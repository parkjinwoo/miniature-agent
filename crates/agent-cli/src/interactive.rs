use std::env;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use agent_model::{LlmMessage, LlmRole, MessagePart, TextPart};
use agent_core::Agent;
use agent_session::{CompactionResult, SessionStore};
use agent_tui::{PromptAction, RunningAction, TuiApp};
use secrecy::SecretString;
use tokio::sync::{mpsc, Mutex};

use crate::paths::AppPaths;
use crate::provider_registry::ProviderSpec;
use crate::session_ui::{
    SessionSelectionOutcome, format_session_summary_line, resolve_selected_session,
};
use crate::compaction::{compact_session, maybe_auto_compact};
use crate::runtime::AppBackend;

pub(crate) struct InteractiveConfig<'a> {
    pub session_dir: &'a Path,
    pub app_paths: &'a AppPaths,
    pub provider_spec: &'a ProviderSpec,
    pub compact_keep_last: usize,
    pub auto_compact_threshold: usize,
    pub auto_compact_keep_last: usize,
    pub api_key: &'a str,
    pub backend: &'a AppBackend,
    pub summary_model: &'a str,
}

pub(crate) async fn run_interactive(
    session: &mut SessionStore,
    agent: Arc<Mutex<Agent<AppBackend>>>,
    config: InteractiveConfig<'_>,
) -> anyhow::Result<()> {
    {
        let mut guard = agent.lock().await;
        guard.set_state(agent_core::AgentState {
            messages: session.messages(),
        });
    }

    let mut tui = TuiApp::new()?;
    tui.replace_messages(&session.messages());
    apply_footer_context(&mut tui, session, config.provider_spec);
    tui.set_status("Ready");
    append_provider_mismatch_note(session, config.provider_spec, &mut tui);
    tui.enter()?;
    let mut queued_input: Option<String> = None;

    loop {
        set_prompt_status(&mut tui, session, config.provider_spec);
        let input = if let Some(input) = queued_input.take() {
            Some(input)
        } else {
            loop {
                match tui.poll_prompt_action(Duration::from_millis(50))? {
                    Some(PromptAction::Submit(input)) => break Some(input),
                    Some(PromptAction::Quit) => {
                        break None;
                    }
                    Some(PromptAction::Continue) | None => {}
                }
            }
        };
        let Some(input) = input else {
            break;
        };

        tui.push_user_input(&input);

        if try_handle_local_command(
            &input,
            session,
            &agent,
            &mut tui,
            &config,
        )
        .await?
        {
            continue;
        }

        let snapshot = {
            let guard = agent.lock().await;
            guard.state().clone()
        };
        let (tx, mut rx) = mpsc::unbounded_channel();
        tui.set_status("Running");
        tui.redraw()?;

        let agent_task = Arc::clone(&agent);
        let api_key_value = config.api_key.to_string();
        let handle = tokio::spawn(async move {
            let mut guard = agent_task.lock().await;
            guard
                .prompt_with(
                    LlmMessage {
                        role: LlmRole::User,
                        parts: vec![MessagePart::Text(TextPart { text: input })],
                    },
                    SecretString::new(api_key_value.into()),
                    |event: &agent_core::AgentEvent| {
                        let _ = tx.send(event.clone());
                    },
                )
                .await
        });

        let mut aborted = false;
        let mut failed = false;
        let result = loop {
            while let Ok(event) = rx.try_recv() {
                tui.push_event(event);
            }
            tui.redraw()?;

            if handle.is_finished() {
                match handle.await {
                    Ok(Ok(run)) => break run,
                    Ok(Err(error)) => {
                        let mut guard = agent.lock().await;
                        guard.set_state(snapshot.clone());
                        tui.push_system_note(format!("run failed: {error}"));
                        tui.set_status("Run failed");
                        tui.redraw()?;
                        failed = true;
                        break agent_core::AgentRunResult::default();
                    }
                    Err(error) => {
                        let mut guard = agent.lock().await;
                        guard.set_state(snapshot.clone());
                        tui.push_system_note(format!("run task failed: {error}"));
                        tui.set_status("Run failed");
                        tui.redraw()?;
                        failed = true;
                        break agent_core::AgentRunResult::default();
                    }
                }
            }

            match tui.poll_running_action(Duration::from_millis(50))? {
                RunningAction::Abort => {
                    handle.abort();
                    let mut guard = agent.lock().await;
                    guard.set_state(snapshot.clone());
                    tui.push_system_note("run aborted");
                    tui.set_status("Aborted");
                    tui.redraw()?;
                    aborted = true;
                    break agent_core::AgentRunResult::default();
                }
                RunningAction::Quit => {
                    handle.abort();
                    let mut guard = agent.lock().await;
                    guard.set_state(snapshot.clone());
                    tui.leave()?;
                    return Ok(());
                }
                RunningAction::QueueSubmit(input) => {
                    queued_input = Some(input);
                    tui.set_status("Queued next prompt");
                }
                RunningAction::Continue => {}
            }
        };

        if aborted {
            continue;
        }

        if failed {
            continue;
        }

        session.append_run(&result)?;
        if let Some(compaction) = maybe_auto_compact(
            session,
            config.auto_compact_threshold,
            config.auto_compact_keep_last,
            Some(config.api_key.to_string()),
            config.backend,
            config.summary_model,
        )
        .await?
        {
            reset_agent_and_tui(
                session,
                &agent,
                &mut tui,
                config.provider_spec,
                Some(&compaction),
            )
            .await?;
        } else {
            tui.set_status(config.provider_spec.display_name);
        }
        tui.redraw()?;
    }

    tui.leave()
}

async fn try_handle_local_command(
    input: &str,
    session: &mut SessionStore,
    agent: &Arc<Mutex<Agent<AppBackend>>>,
    tui: &mut TuiApp,
    config: &InteractiveConfig<'_>,
) -> anyhow::Result<bool> {
    let trimmed = input.trim();
    if !trimmed.starts_with('/') {
        return Ok(false);
    }

    let parts = trimmed.split_whitespace().collect::<Vec<_>>();
    match parts.first().copied() {
        Some("/compact") => {
            let keep_last = parts
                .get(1)
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(config.compact_keep_last);
            match compact_session(
                session,
                keep_last,
                Some(config.api_key.to_string()),
                config.backend,
                config.summary_model,
            )
            .await?
            {
                Some(compaction) => {
                    reset_agent_and_tui(
                        session,
                        agent,
                        tui,
                        config.provider_spec,
                        Some(&compaction),
                    )
                    .await?;
                    tui.push_system_note(format!(
                        "compacted {} messages into {}",
                        compaction.compacted_message_count,
                        compaction.summary_entry_id
                    ));
                }
                None => {
                    tui.push_system_note(format!(
                        "compaction skipped: keep_last={} leaves too little to compact",
                        keep_last
                    ));
                    tui.set_status("Compaction skipped");
                }
            }
            tui.redraw()?;
            Ok(true)
        }
        Some("/sessions") => {
            let sessions = crate::session_ui::prioritize_matching_sessions(
                SessionStore::list_sessions(config.session_dir)?,
                config.provider_spec,
            )
            .into_iter()
            .collect::<Vec<_>>();
            let items = sessions
                .iter()
                .map(|summary| format_session_summary_line(summary, Some(config.provider_spec)))
                .collect::<Vec<_>>();
            if items.is_empty() {
                tui.push_system_note("no sessions found");
                tui.set_status("No sessions");
            } else {
                let initial_index = sessions
                    .iter()
                    .position(|summary| summary.path == session.path())
                    .unwrap_or(0);
                if let Some(index) =
                    tui.pick_from_list_at(
                        "Select session (= match, ! mismatch, ? unknown)",
                        &items,
                        initial_index,
                    )?
            {
                let Some((next_session, outcome)) = resolve_selected_session(
                    tui,
                    config.session_dir,
                    &sessions[index],
                    config.provider_spec,
                )? else {
                    tui.set_status("Session switch cancelled");
                    tui.redraw()?;
                    return Ok(true);
                };
                *session = next_session;
                reset_agent_and_tui(session, agent, tui, config.provider_spec, None).await?;
                match outcome {
                    SessionSelectionOutcome::OpenedMatch => {
                        tui.push_system_note(format!("opened session {}", session.path().display()));
                    }
                    SessionSelectionOutcome::OpenedMismatch => {
                        tui.push_system_note(format!(
                            "opened mismatched session {}",
                            session.path().display()
                        ));
                        append_provider_mismatch_note(session, config.provider_spec, tui);
                    }
                    SessionSelectionOutcome::ForkedMismatch => {
                        tui.push_system_note(format!(
                            "forked mismatched session into {} with provider {}",
                            session.path().display(),
                            config.provider_spec.display_name
                        ));
                        tui.set_status("Forked from mismatched session");
                    }
                }
            }
            }
            tui.redraw()?;
            Ok(true)
        }
        Some("/tree") => {
            let checkpoints = session.checkpoints();
            if checkpoints.is_empty() {
                tui.push_system_note("no checkpoints in current session");
                tui.set_status("No checkpoints");
                tui.redraw()?;
                return Ok(true);
            }

            let checkpoint_items = checkpoints.iter().rev().collect::<Vec<_>>();
            let items = checkpoint_items
                .iter()
                .map(|checkpoint| {
                    let branch_marker = if checkpoint.is_current_leaf { "●" } else { "·" };
                    let label = format_tree_checkpoint_label(session, checkpoint);
                    format!(
                        "{branch_marker} {} {}",
                        format_short_timestamp(&checkpoint.timestamp),
                        label
                    )
                })
                .collect::<Vec<_>>();
            let initial_index = checkpoint_items
                .iter()
                .position(|checkpoint| checkpoint.is_current_leaf)
                .unwrap_or(0);
            if let Some(index) = tui.pick_from_list_at("Select checkpoint", &items, initial_index)? {
                session.set_leaf(Some(checkpoint_items[index].entry_id.clone()));
                reset_agent_and_tui(session, agent, tui, config.provider_spec, None).await?;
                tui.push_system_note(format!(
                    "moved to checkpoint {}",
                    checkpoint_items[index].entry_id
                ));
            }
            tui.redraw()?;
            Ok(true)
        }
        Some("/fork") => {
            *session = session.fork_with_provider(
                config.session_dir,
                Some(config.provider_spec.to_session_info()),
            )?;
            reset_agent_and_tui(session, agent, tui, config.provider_spec, None).await?;
            tui.push_system_note(format!(
                "forked current session into {} with provider {}",
                session.path().display(),
                config.provider_spec.display_name
            ));
            tui.set_status("Forked session");
            tui.redraw()?;
            Ok(true)
        }
        Some("/provider") => {
            let mut lines = config.provider_spec.describe_lines();
            lines.push(format!(
                "session_header_provider: {}",
                session
                    .header()
                    .provider
                    .as_ref()
                    .map(|provider| provider.display_name.as_str())
                    .unwrap_or("-")
            ));
            if let Some(session_provider) = session.header().provider.as_ref() {
                let mismatches = config.provider_spec.mismatch_lines(session_provider);
                if mismatches.is_empty() {
                    lines.push("session_match: yes".to_string());
                } else {
                    lines.push("session_match: no".to_string());
                    lines.extend(mismatches.into_iter().map(|line| format!("diff: {line}")));
                    lines.push("recommended_next_step: /fork".to_string());
                }
            } else {
                lines.push("session_match: unknown".to_string());
            }
            tui.push_system_note(lines.join("\n"));
            tui.set_status("Provider info");
            tui.redraw()?;
            Ok(true)
        }
        Some("/paths") => {
            tui.push_system_note(format!(
                "config_file: {}\nstate_dir: {}\nsessions_dir: {}",
                config.app_paths.config_file.display(),
                config.app_paths.state_dir.display(),
                config.session_dir.display()
            ));
            tui.set_status("Paths");
            tui.redraw()?;
            Ok(true)
        }
        Some("/help") => {
            tui.push_system_note(
                "local commands: /compact [keep_last], /sessions, /tree, /fork, /provider, /paths, /help",
            );
            tui.set_status("Help");
            tui.redraw()?;
            Ok(true)
        }
        Some(other) => {
            tui.push_system_note(format!("unknown local command: {other}"));
            tui.set_status("Unknown command");
            tui.redraw()?;
            Ok(true)
        }
        None => Ok(false),
    }
}

async fn reset_agent_and_tui(
    session: &mut SessionStore,
    agent: &Arc<Mutex<Agent<AppBackend>>>,
    tui: &mut TuiApp,
    provider_spec: &ProviderSpec,
    compaction: Option<&CompactionResult>,
) -> anyhow::Result<()> {
    let messages = session.messages();
    let mut guard = agent.lock().await;
    guard.set_state(agent_core::AgentState {
        messages: messages.clone(),
    });
    drop(guard);

    tui.replace_messages(&messages);
    apply_footer_context(tui, session, provider_spec);
    if let Some(compaction) = compaction {
        tui.set_status(format!(
            "{} · compacted {}",
            provider_spec.display_name,
            compaction.compacted_message_count,
        ));
    } else {
        tui.set_status("Ready");
    }
    Ok(())
}

fn append_provider_mismatch_note(
    session: &SessionStore,
    provider_spec: &ProviderSpec,
    tui: &mut TuiApp,
) {
    let Some(session_provider) = session.header().provider.as_ref() else {
        return;
    };

    let mismatches = provider_spec.mismatch_lines(session_provider);
    if mismatches.is_empty() {
        return;
    }

    let session_provider_name = session_provider.display_name.as_str();
    tui.push_system_note(format!(
        "provider/session mismatch detected:\nruntime provider: {}\nsession provider: {}\n{}\nrecommended next step: run /fork before continuing so this runtime configuration branches from the existing transcript.",
        provider_spec.display_name,
        session_provider_name,
        mismatches.join("\n")
    ));
    tui.set_status("Provider mismatch, /fork recommended");
}

fn set_prompt_status(tui: &mut TuiApp, session: &SessionStore, provider_spec: &ProviderSpec) {
    if let Some(session_provider) = session.header().provider.as_ref() {
        if !provider_spec.mismatch_lines(session_provider).is_empty() {
            tui.set_status("Provider mismatch, /fork recommended");
            return;
        }
    }

    tui.set_status("Ready");
}

fn apply_footer_context(
    tui: &mut TuiApp,
    session: &SessionStore,
    provider_spec: &ProviderSpec,
) {
    let cwd = Path::new(&session.header().cwd);
    let path_label = format_home_path(cwd);
    tui.set_footer_context(path_label, provider_spec.default_model.clone());
}

fn format_home_path(path: &Path) -> String {
    let path_str = path.to_string_lossy();
    let Ok(home) = env::var("HOME") else {
        return path_str.into_owned();
    };
    let home_path = Path::new(&home);
    if path == home_path {
        return "~".to_string();
    }
    if let Ok(relative) = path.strip_prefix(home_path) {
        let relative_str = relative.to_string_lossy();
        if relative_str.is_empty() {
            "~".to_string()
        } else {
            format!("~/{}", relative_str)
        }
    } else {
        path_str.into_owned()
    }
}

fn format_short_timestamp(timestamp: &str) -> String {
    timestamp
        .get(2..19)
        .map(|value| value.replace('T', " "))
        .unwrap_or_else(|| timestamp.to_string())
}

fn format_tree_checkpoint_label(
    session: &SessionStore,
    checkpoint: &agent_session::SessionCheckpoint,
) -> String {
    let raw = checkpoint
        .label
        .lines()
        .next()
        .unwrap_or_default()
        .trim();

    if let Some(rest) = raw.strip_prefix("You: ") {
        return format!("Me: {rest}");
    }

    if let Some(rest) = raw.strip_prefix("AI: ") {
        let model = session
            .header()
            .provider
            .as_ref()
            .map(|provider| provider.model.as_str())
            .unwrap_or("Assistant");
        return format!("{model}: {rest}");
    }

    raw.to_string()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::{format_home_path, format_short_timestamp, format_tree_checkpoint_label};
    use agent_model::{LlmMessage, LlmRole, MessagePart, TextPart};
    use agent_session::{SessionProviderInfo, SessionStore};
    use uuid::Uuid;

    #[test]
    fn format_home_path_rewrites_paths_under_home() {
        let home = std::env::var("HOME").unwrap();
        let nested = Path::new(&home).join("work/miniature-agent");
        assert_eq!(format_home_path(&nested), "~/work/miniature-agent");
    }

    #[test]
    fn format_home_path_keeps_paths_outside_home() {
        let path = Path::new("/tmp/example");
        assert_eq!(format_home_path(path), "/tmp/example");
    }

    #[test]
    fn format_short_timestamp_prefers_compact_full_datetime() {
        assert_eq!(
            format_short_timestamp("2026-04-03T12:34:56+09:00"),
            "26-04-03 12:34:56"
        );
        assert_eq!(format_short_timestamp("short"), "short");
    }

    #[test]
    fn tree_checkpoint_label_uses_me_and_model_name() {
        let temp = std::env::temp_dir().join(format!("miniature-agent-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&temp).unwrap();
        let mut store = SessionStore::create(
            &temp,
            "/workspace",
            None,
            Some(SessionProviderInfo {
                display_name: "Compatible".to_string(),
                model: "gpt-oss".to_string(),
                backend: "chat-completions:compatible".to_string(),
                resolved_base_url: None,
                compat: None,
            }),
        )
        .unwrap();
        let user = agent_core::AgentMessage::User(LlmMessage {
            role: LlmRole::User,
            parts: vec![MessagePart::Text(TextPart {
                text: "hello".to_string(),
            })],
        });
        let assistant = agent_core::AgentMessage::Assistant(LlmMessage {
            role: LlmRole::Assistant,
            parts: vec![MessagePart::Text(TextPart {
                text: "world".to_string(),
            })],
        });
        store.append_run(&agent_core::AgentRunResult {
            new_messages: vec![user.clone(), assistant.clone()],
            ..Default::default()
        }).unwrap();
        let checkpoints = store.checkpoints();
        let user_label = format_tree_checkpoint_label(&store, &checkpoints[0]);
        let assistant_label = format_tree_checkpoint_label(&store, &checkpoints[1]);
        assert_eq!(user_label, "Me: hello");
        assert_eq!(assistant_label, "gpt-oss: world");
        let _ = fs::remove_dir_all(&temp);
    }
}

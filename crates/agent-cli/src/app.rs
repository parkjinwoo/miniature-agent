use std::sync::Arc;

use agent_core::{Agent, AgentConfig};
use agent_tools::default_tool_registry;
use tokio::sync::Mutex;

use crate::args::{parse_model, parse_prompt, parse_provider, print_paths};
use crate::bootstrap::{list_sessions, open_session_from_args, resolve_api_key};
use crate::compaction::compact_session;
use crate::config::{write_example_config, AppConfig};
use crate::interactive::{run_interactive, InteractiveConfig};
use crate::non_interactive::{run_prompt, NonInteractiveConfig};
use crate::paths::AppPaths;
use crate::provider_registry::Provider;
use crate::runtime::configured_backend;

const COMPACT_KEEP_LAST: usize = 8;
const AUTO_COMPACT_THRESHOLD: usize = 24;
const AUTO_COMPACT_KEEP_LAST: usize = COMPACT_KEEP_LAST;

pub(crate) async fn run(args: Vec<String>) -> anyhow::Result<()> {
    let compact_requested = args.iter().any(|arg| arg == "--compact");
    let app_paths = AppPaths::discover()?;
    let app_config = AppConfig::load(&app_paths)?;
    let cwd = std::env::current_dir()?;
    if args.iter().any(|arg| arg == "--print-paths") {
        print_paths(&app_paths, &app_config);
        return Ok(());
    }
    if args.iter().any(|arg| arg == "--write-default-config") {
        write_example_config(&app_paths.config_file)?;
        println!("Wrote {}", app_paths.config_file.display());
        return Ok(());
    }

    app_paths.ensure_directories()?;
    let provider = parse_provider(&args).unwrap_or_else(|| infer_default_provider(&app_config));
    let mut provider_spec = provider.load_spec(&app_config)?;
    let model = parse_model(&args).unwrap_or_else(|| provider_spec.default_model.clone());
    let summary_model = provider_spec.default_summary_model.clone();
    provider_spec.default_model = model.clone();
    let runtime_provider = provider_spec.to_session_info();
    let prompt = parse_prompt(&args);
    let session_dir = app_config.resolved_session_dir(&app_paths);

    if args.iter().any(|arg| arg == "--list-sessions") {
        list_sessions(&session_dir, &provider_spec)?;
        return Ok(());
    }

    let backend = configured_backend(&provider_spec);
    let tools = default_tool_registry(cwd.clone());
    let mut agent = Agent::new(
        backend.clone(),
        AgentConfig {
            system: "You are a coding agent.".to_string(),
            model,
            temperature: None,
        },
        tools,
    );

    let Some(mut session) = open_session_from_args(&args, &session_dir, &cwd, &runtime_provider)? else {
        return Ok(());
    };
    let provider_api_key = resolve_api_key(&provider_spec);
    if compact_requested {
        match compact_session(
            &mut session,
            COMPACT_KEEP_LAST,
            provider_api_key.clone(),
            &backend,
            &summary_model,
        )
        .await?
        {
            Some(result) => println!(
                "Compacted {} messages into summary {}. New leaf={}",
                result.compacted_message_count,
                result.summary_entry_id,
                result.leaf_id.as_deref().unwrap_or("-"),
            ),
            None => println!(
                "Compaction skipped: not enough messages to compact with keep_last={COMPACT_KEEP_LAST}."
            ),
        }
    }

    let Some(api_key) = provider_api_key else {
        if compact_requested {
            return Ok(());
        }
        println!(
            "Set {} to run the {} smoke CLI.",
            provider_spec.api_key_env,
            provider_spec.display_name
        );
        return Ok(());
    };

    if let Some(prompt) = prompt {
        run_prompt(
            &mut session,
            &mut agent,
            NonInteractiveConfig {
                prompt,
                api_key: &api_key,
                auto_compact_threshold: AUTO_COMPACT_THRESHOLD,
                auto_compact_keep_last: AUTO_COMPACT_KEEP_LAST,
                backend: &backend,
                summary_model: &summary_model,
            },
        )
        .await?;
        return Ok(());
    }

    let agent = Arc::new(Mutex::new(agent));
    run_interactive(
        &mut session,
        agent,
        InteractiveConfig {
            session_dir: &session_dir,
            app_paths: &app_paths,
            provider_spec: &provider_spec,
            compact_keep_last: COMPACT_KEEP_LAST,
            auto_compact_threshold: AUTO_COMPACT_THRESHOLD,
            auto_compact_keep_last: AUTO_COMPACT_KEEP_LAST,
            api_key: &api_key,
            backend: &backend,
            summary_model: &summary_model,
        },
    )
    .await
}

pub(crate) fn infer_default_provider(config: &AppConfig) -> Provider {
    let providers = [Provider::Compatible, Provider::Anthropic, Provider::OpenAi];

    let configured = providers
        .into_iter()
        .filter(|provider| config.provider_override(provider.key()).is_some())
        .collect::<Vec<_>>();
    if configured.len() == 1 {
        return configured[0];
    }

    let providers = [Provider::Compatible, Provider::Anthropic, Provider::OpenAi];
    let env_backed = providers
        .into_iter()
        .filter(|provider| provider_has_runtime_env(*provider, config))
        .collect::<Vec<_>>();
    if env_backed.len() == 1 {
        return env_backed[0];
    }

    Provider::OpenAi
}

fn provider_has_runtime_env(provider: Provider, config: &AppConfig) -> bool {
    let Ok(spec) = provider.load_spec(config) else {
        return false;
    };

    std::env::var(&spec.api_key_env)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .is_some()
        || spec
            .base_url_env
            .as_deref()
            .and_then(|name| std::env::var(name).ok())
            .filter(|value| !value.trim().is_empty())
            .is_some()
}

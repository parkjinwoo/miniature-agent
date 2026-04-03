use std::path::Path;

use agent_session::SessionStore;
use agent_tui::TuiApp;

use crate::provider_registry::ProviderSpec;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SessionSelectionOutcome {
    OpenedMatch,
    OpenedMismatch,
    ForkedMismatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MismatchResolution {
    Fork,
    OpenExisting,
    Cancel,
}

pub(crate) fn resolve_selected_session(
    tui: &mut TuiApp,
    session_dir: &Path,
    summary: &agent_session::SessionSummary,
    provider_spec: &ProviderSpec,
) -> anyhow::Result<Option<(SessionStore, SessionSelectionOutcome)>> {
    if session_matches_provider(summary, provider_spec) {
        return SessionStore::open(&summary.path)
            .map(|store| Some((store, SessionSelectionOutcome::OpenedMatch)));
    }

    let prompt = mismatch_resolution_prompt(summary, provider_spec);
    let options = mismatch_resolution_options(provider_spec);

    match pick_mismatch_resolution(tui, &prompt, &options)? {
        MismatchResolution::Fork => {
            let store = SessionStore::open(&summary.path)?;
            let forked = store.fork_with_provider(session_dir, Some(provider_spec.to_session_info()))?;
            Ok(Some((forked, SessionSelectionOutcome::ForkedMismatch)))
        }
        MismatchResolution::OpenExisting => SessionStore::open(&summary.path)
            .map(|store| Some((store, SessionSelectionOutcome::OpenedMismatch))),
        MismatchResolution::Cancel => Ok(None),
    }
}

fn pick_mismatch_resolution(
    tui: &mut TuiApp,
    prompt: &str,
    options: &[String],
) -> anyhow::Result<MismatchResolution> {
    Ok(match tui.pick_from_list(prompt, options)? {
        Some(0) => MismatchResolution::Fork,
        Some(1) => MismatchResolution::OpenExisting,
        Some(_) | None => MismatchResolution::Cancel,
    })
}

pub(crate) fn mismatch_resolution_prompt(
    summary: &agent_session::SessionSummary,
    provider_spec: &ProviderSpec,
) -> String {
    let mismatch_details = mismatch_summary_lines(summary, provider_spec);
    format!(
        "Session provider differs\nruntime={}\nsession={}\n{}\nrecommended: use /fork or choose the first option to keep this transcript and continue on a clean branch.",
        runtime_provider_label(provider_spec),
        session_provider_label(summary),
        mismatch_details.join("\n")
    )
}

pub(crate) fn mismatch_resolution_options(provider_spec: &ProviderSpec) -> Vec<String> {
    vec![
        format!("Fork with current provider ({}) [recommended]", provider_spec.display_name),
        "Open existing session as-is [advanced]".to_string(),
        "Cancel".to_string(),
    ]
}

pub(crate) fn format_session_summary_line(
    session: &agent_session::SessionSummary,
    runtime_provider: Option<&ProviderSpec>,
) -> String {
    let marker = match runtime_provider {
        Some(provider_spec) if session_matches_provider(session, provider_spec) => "=",
        Some(_) if session.provider_display_name.is_some() => "!",
        Some(_) => "?",
        None => "-",
    };
    let mismatch_suffix = runtime_provider
        .filter(|provider_spec| !session_matches_provider(session, provider_spec))
        .map(|provider_spec| format!("  diff={}", mismatch_summary_inline(session, provider_spec)))
        .unwrap_or_default();
    let session_name = session
        .path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("session");
    format!(
        "{} {}  {}  {}{}",
        marker,
        format_short_timestamp(&session.created_at),
        session_provider_label(session),
        session_name,
        mismatch_suffix
    )
}

pub(crate) fn session_matches_provider(
    session: &agent_session::SessionSummary,
    provider_spec: &ProviderSpec,
) -> bool {
    let Some(display_name) = session.provider_display_name.as_deref() else {
        return false;
    };
    let Some(backend) = session.provider_backend.as_deref() else {
        return false;
    };
    let Some(model) = session.provider_model.as_deref() else {
        return false;
    };

    let runtime = provider_spec.to_session_info();
    display_name == runtime.display_name && backend == runtime.backend && model == runtime.model
}

pub(crate) fn mismatch_summary_lines(
    session: &agent_session::SessionSummary,
    provider_spec: &ProviderSpec,
) -> Vec<String> {
    let mut lines = Vec::new();

    let runtime = provider_spec.to_session_info();
    if session.provider_backend.as_deref() != Some(runtime.backend.as_str()) {
        lines.push(format!(
            "backend: runtime={} session={}",
            runtime.backend,
            session.provider_backend.as_deref().unwrap_or("-")
        ));
    }
    if session.provider_model.as_deref() != Some(runtime.model.as_str()) {
        lines.push(format!(
            "model: runtime={} session={}",
            runtime.model,
            session.provider_model.as_deref().unwrap_or("-")
        ));
    }
    if session.provider_display_name.as_deref() != Some(runtime.display_name.as_str()) {
        lines.push(format!(
            "provider: runtime={} session={}",
            runtime.display_name,
            session.provider_display_name.as_deref().unwrap_or("-")
        ));
    }

    if lines.is_empty() {
        lines.push("provider snapshot differs".to_string());
    }

    lines
}

pub(crate) fn mismatch_summary_inline(
    session: &agent_session::SessionSummary,
    provider_spec: &ProviderSpec,
) -> String {
    mismatch_summary_lines(session, provider_spec)
        .into_iter()
        .map(|line| {
            line.replace("backend: runtime=", "backend ")
                .replace("model: runtime=", "model ")
                .replace("provider: runtime=", "provider ")
                .replace(" session=", " ≠ ")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

pub(crate) fn session_provider_label(session: &agent_session::SessionSummary) -> String {
    let provider = session.provider_display_name.as_deref().unwrap_or("-");
    let model = session.provider_model.as_deref().unwrap_or("-");
    format!("{provider}/{model}")
}

pub(crate) fn runtime_provider_label(provider_spec: &ProviderSpec) -> String {
    format!("{}/{}", provider_spec.display_name, provider_spec.default_model)
}

fn format_short_timestamp(timestamp: &str) -> String {
    timestamp
        .get(2..19)
        .map(|value| value.replace('T', " "))
        .unwrap_or_else(|| timestamp.to_string())
}

pub(crate) fn prioritize_matching_sessions(
    mut sessions: Vec<agent_session::SessionSummary>,
    provider_spec: &ProviderSpec,
) -> Vec<agent_session::SessionSummary> {
    sessions.sort_by_key(|session| {
        if session_matches_provider(session, provider_spec) {
            0
        } else {
            1
        }
    });
    sessions
}

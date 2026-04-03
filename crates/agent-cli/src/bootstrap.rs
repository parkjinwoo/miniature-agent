use std::path::Path;

use agent_session::{SessionProviderInfo, SessionStore};

use crate::provider_registry::ProviderSpec;
use crate::session_ui::format_session_summary_line;

pub(crate) fn list_sessions(
    session_dir: &Path,
    provider_spec: &ProviderSpec,
) -> anyhow::Result<()> {
    for session in SessionStore::list_sessions(session_dir)? {
        println!(
            "{}",
            format_session_summary_line(&session, Some(provider_spec))
        );
    }
    Ok(())
}

pub(crate) fn open_session_from_args(
    args: &[String],
    session_dir: &Path,
    cwd: &Path,
    runtime_provider: &SessionProviderInfo,
) -> anyhow::Result<Option<SessionStore>> {
    let cwd_display = cwd.display().to_string();
    let session = if args.iter().any(|arg| arg == "--new-session") {
        SessionStore::create(
            session_dir,
            cwd_display,
            None,
            Some(runtime_provider.clone()),
        )?
    } else {
        SessionStore::open_or_create_latest(
            session_dir,
            cwd.display().to_string(),
            Some(runtime_provider.clone()),
        )?
    };

    Ok(Some(session))
}

pub(crate) fn resolve_api_key(provider_spec: &ProviderSpec) -> Option<String> {
    std::env::var(&provider_spec.api_key_env)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            if provider_spec.requires_api_key {
                None
            } else {
                Some(String::new())
            }
        })
}

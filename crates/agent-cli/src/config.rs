use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Deserialize;

use crate::paths::AppPaths;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct AppConfig {
    pub session_dir: Option<PathBuf>,
    #[serde(default)]
    pub providers: HashMap<String, ProviderOverrideConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ProviderOverrideConfig {
    pub display_name: Option<String>,
    pub api_key_env: Option<String>,
    pub requires_api_key: Option<bool>,
    pub default_model: Option<String>,
    pub default_summary_model: Option<String>,
    pub base_url_env: Option<String>,
    pub base_url: Option<String>,
    pub compat: Option<CompatOverrideConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct CompatOverrideConfig {
    pub supports_reasoning_effort: Option<bool>,
    pub supports_developer_role: Option<bool>,
    pub requires_tool_result_name: Option<bool>,
    pub reasoning_field: Option<String>,
}

impl AppConfig {
    pub fn load(paths: &AppPaths) -> anyhow::Result<Self> {
        if !paths.config_file.exists() {
            return Ok(Self::default());
        }

        let raw = std::fs::read_to_string(&paths.config_file)
            .with_context(|| format!("failed to read {}", paths.config_file.display()))?;
        toml::from_str(&raw)
            .with_context(|| format!("failed to parse {}", paths.config_file.display()))
    }

    pub fn resolved_session_dir(&self, paths: &AppPaths) -> PathBuf {
        match &self.session_dir {
            Some(path) if path.is_absolute() => path.clone(),
            Some(path) => paths.config_dir.join(path),
            None => paths.sessions_dir.clone(),
        }
    }

    pub fn provider_override(&self, provider_key: &str) -> Option<&ProviderOverrideConfig> {
        self.providers.get(provider_key)
    }
}

pub fn write_example_config(path: &Path) -> anyhow::Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("config path has no parent: {}", path.display()))?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("failed to create {}", parent.display()))?;
    std::fs::write(path, DEFAULT_CONFIG_TEMPLATE)
        .with_context(|| format!("failed to write {}", path.display()))
}

pub const DEFAULT_CONFIG_TEMPLATE: &str = r#"# miniature-agent config
#
# Standard XDG locations:
# - config: ~/.config/miniature-agent/config.toml
# - state:  ~/.local/state/miniature-agent/sessions/

# session_dir = "/absolute/path/to/sessions"

[providers.openai]
# default_model = "gpt-5"

[providers.compatible]
# default_model = "your-model"
# base_url = "http://localhost:11434/v1/chat/completions"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_relative_session_dir_from_config_dir() {
        let paths = AppPaths {
            config_dir: PathBuf::from("/tmp/miniature-agent-config"),
            config_file: PathBuf::from("/tmp/miniature-agent-config/config.toml"),
            state_dir: PathBuf::from("/tmp/miniature-agent-state"),
            sessions_dir: PathBuf::from("/tmp/miniature-agent-state/sessions"),
        };
        let config = AppConfig {
            session_dir: Some(PathBuf::from("sessions-local")),
            providers: HashMap::new(),
        };

        assert_eq!(
            config.resolved_session_dir(&paths),
            PathBuf::from("/tmp/miniature-agent-config/sessions-local")
        );
    }
}

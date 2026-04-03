use agent_model::ChatCompletionsCompat;
use agent_session::{SessionProviderCompat, SessionProviderInfo};

use crate::config::{AppConfig, ProviderOverrideConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Provider {
    OpenAi,
    Anthropic,
    Compatible,
}

#[derive(Debug, Clone)]
pub struct ProviderSpec {
    pub display_name: &'static str,
    pub api_key_env: String,
    pub requires_api_key: bool,
    pub default_model: String,
    pub default_summary_model: String,
    pub base_url: Option<String>,
    pub base_url_env: Option<String>,
    pub backend: BackendSpec,
}

#[derive(Debug, Clone)]
pub enum BackendSpec {
    OpenAiResponses,
    AnthropicMessages,
    ChatCompletions {
        backend_name: &'static str,
        default_base_url: String,
        compat: ChatCompletionsCompat,
    },
}

impl Provider {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "openai" => Some(Self::OpenAi),
            "anthropic" => Some(Self::Anthropic),
            "compatible" => Some(Self::Compatible),
            _ => None,
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
            Self::Compatible => "compatible",
        }
    }

    pub fn load_spec(self, config: &AppConfig) -> anyhow::Result<ProviderSpec> {
        let mut spec = self.default_spec();
        if let Some(overrides) = config.provider_override(self.key()) {
            spec.apply_override(overrides);
        }
        Ok(spec)
    }

    fn default_spec(self) -> ProviderSpec {
        match self {
            Self::OpenAi => ProviderSpec {
                display_name: "OpenAI",
                api_key_env: "MINIATURE_AGENT_OPENAI_API_KEY".to_string(),
                requires_api_key: true,
                default_model: "gpt-5".to_string(),
                default_summary_model: "gpt-5".to_string(),
                base_url: None,
                base_url_env: Some("MINIATURE_AGENT_OPENAI_BASE_URL".to_string()),
                backend: BackendSpec::OpenAiResponses,
            },
            Self::Anthropic => ProviderSpec {
                display_name: "Anthropic",
                api_key_env: "MINIATURE_AGENT_ANTHROPIC_API_KEY".to_string(),
                requires_api_key: true,
                default_model: "claude-sonnet-4-20250514".to_string(),
                default_summary_model: "claude-sonnet-4-20250514".to_string(),
                base_url: None,
                base_url_env: Some("MINIATURE_AGENT_ANTHROPIC_BASE_URL".to_string()),
                backend: BackendSpec::AnthropicMessages,
            },
            Self::Compatible => ProviderSpec {
                display_name: "Compatible",
                api_key_env: "MINIATURE_AGENT_COMPATIBLE_API_KEY".to_string(),
                requires_api_key: false,
                default_model: "your-model".to_string(),
                default_summary_model: "your-model".to_string(),
                base_url: None,
                base_url_env: Some("MINIATURE_AGENT_COMPATIBLE_BASE_URL".to_string()),
                backend: BackendSpec::ChatCompletions {
                    backend_name: "compatible",
                    default_base_url: "http://localhost:11434/v1/chat/completions".to_string(),
                    compat: ChatCompletionsCompat {
                        supports_reasoning_effort: false,
                        supports_developer_role: false,
                        requires_tool_result_name: false,
                        reasoning_field: None,
                    },
                },
            },
        }
    }
}

impl ProviderSpec {
    pub fn resolved_base_url(&self) -> Option<String> {
        self.base_url.clone().or_else(|| {
            self.base_url_env.as_deref().and_then(|env_name| match std::env::var(env_name) {
                Ok(base_url) if !base_url.trim().is_empty() => Some(base_url),
                _ => None,
            })
        })
    }

    pub fn describe_lines(&self) -> Vec<String> {
        let mut lines = vec![
            format!("provider: {}", self.display_name),
            format!("api_key_env: {}", self.api_key_env),
            format!("requires_api_key: {}", self.requires_api_key),
            format!("model: {}", self.default_model),
            format!("summary_model: {}", self.default_summary_model),
        ];

        if let Some(base_url_env) = &self.base_url_env {
            lines.push(format!("base_url_env: {}", base_url_env));
        }
        if let Some(base_url) = &self.base_url {
            lines.push(format!("configured_base_url: {}", base_url));
        }

        match &self.backend {
            BackendSpec::OpenAiResponses => {
                lines.push("backend: openai-responses".to_string());
            }
            BackendSpec::AnthropicMessages => {
                lines.push("backend: anthropic-messages".to_string());
            }
            BackendSpec::ChatCompletions {
                backend_name,
                default_base_url,
                compat,
            } => {
                lines.push(format!("backend: chat-completions ({backend_name})"));
                lines.push(format!("default_base_url: {}", default_base_url));
                lines.push(format!(
                    "compat: developer_role={} reasoning_effort={} tool_result_name={} reasoning_field={}",
                    compat.supports_developer_role,
                    compat.supports_reasoning_effort,
                    compat.requires_tool_result_name,
                    compat.reasoning_field.as_deref().unwrap_or("-"),
                ));
            }
        }

        if let Some(resolved_base_url) = self.resolved_base_url() {
            lines.push(format!("resolved_base_url: {}", resolved_base_url));
        }

        lines
    }

    pub fn to_session_info(&self) -> SessionProviderInfo {
        match &self.backend {
            BackendSpec::OpenAiResponses => SessionProviderInfo {
                display_name: self.display_name.to_string(),
                model: self.default_model.clone(),
                backend: "openai-responses".to_string(),
                resolved_base_url: self.resolved_base_url(),
                compat: None,
            },
            BackendSpec::AnthropicMessages => SessionProviderInfo {
                display_name: self.display_name.to_string(),
                model: self.default_model.clone(),
                backend: "anthropic-messages".to_string(),
                resolved_base_url: self.resolved_base_url(),
                compat: None,
            },
            BackendSpec::ChatCompletions {
                backend_name,
                compat,
                ..
            } => SessionProviderInfo {
                display_name: self.display_name.to_string(),
                model: self.default_model.clone(),
                backend: format!("chat-completions:{backend_name}"),
                resolved_base_url: self.resolved_base_url(),
                compat: Some(SessionProviderCompat {
                    supports_reasoning_effort: compat.supports_reasoning_effort,
                    supports_developer_role: compat.supports_developer_role,
                    requires_tool_result_name: compat.requires_tool_result_name,
                    reasoning_field: compat.reasoning_field.clone(),
                }),
            },
        }
    }

    pub fn mismatch_lines(&self, session: &SessionProviderInfo) -> Vec<String> {
        let runtime = self.to_session_info();
        let mut lines = Vec::new();

        compare_field(&mut lines, "provider", &runtime.display_name, &session.display_name);
        compare_field(
            &mut lines,
            "model",
            &runtime.model,
            &session.model,
        );
        compare_field(&mut lines, "backend", &runtime.backend, &session.backend);
        compare_optional_field(
            &mut lines,
            "resolved_base_url",
            runtime.resolved_base_url.as_deref(),
            session.resolved_base_url.as_deref(),
        );

        match (&runtime.compat, &session.compat) {
            (Some(runtime_compat), Some(session_compat)) => {
                if runtime_compat.supports_reasoning_effort != session_compat.supports_reasoning_effort {
                    lines.push(format!(
                        "compat.supports_reasoning_effort: runtime={} session={}",
                        runtime_compat.supports_reasoning_effort,
                        session_compat.supports_reasoning_effort
                    ));
                }
                if runtime_compat.supports_developer_role != session_compat.supports_developer_role {
                    lines.push(format!(
                        "compat.supports_developer_role: runtime={} session={}",
                        runtime_compat.supports_developer_role,
                        session_compat.supports_developer_role
                    ));
                }
                if runtime_compat.requires_tool_result_name != session_compat.requires_tool_result_name {
                    lines.push(format!(
                        "compat.requires_tool_result_name: runtime={} session={}",
                        runtime_compat.requires_tool_result_name,
                        session_compat.requires_tool_result_name
                    ));
                }
                compare_optional_field(
                    &mut lines,
                    "compat.reasoning_field",
                    runtime_compat.reasoning_field.as_deref(),
                    session_compat.reasoning_field.as_deref(),
                );
            }
            (Some(_), None) | (None, Some(_)) => {
                lines.push("compat: runtime/session differ".to_string());
            }
            (None, None) => {}
        }

        lines
    }

    fn apply_override(&mut self, override_spec: &ProviderOverrideConfig) {
        if let Some(api_key_env) = &override_spec.api_key_env {
            self.api_key_env = api_key_env.clone();
        }
        if let Some(requires_api_key) = override_spec.requires_api_key {
            self.requires_api_key = requires_api_key;
        }
        if let Some(default_model) = &override_spec.default_model {
            self.default_model = default_model.clone();
        }
        if let Some(default_summary_model) = &override_spec.default_summary_model {
            self.default_summary_model = default_summary_model.clone();
        }
        if let Some(base_url) = &override_spec.base_url {
            self.base_url = Some(base_url.clone());
        }
        if let Some(base_url_env) = &override_spec.base_url_env {
            self.base_url_env = Some(base_url_env.clone());
        }
        if let Some(display_name) = &override_spec.display_name {
            self.display_name = Box::leak(display_name.clone().into_boxed_str());
        }

        if let BackendSpec::ChatCompletions {
            compat,
            ..
        } = &mut self.backend
        {
            if let Some(compat_override) = &override_spec.compat {
                if let Some(value) = compat_override.supports_reasoning_effort {
                    compat.supports_reasoning_effort = value;
                }
                if let Some(value) = compat_override.supports_developer_role {
                    compat.supports_developer_role = value;
                }
                if let Some(value) = compat_override.requires_tool_result_name {
                    compat.requires_tool_result_name = value;
                }
                if compat_override.reasoning_field.is_some() {
                    compat.reasoning_field = compat_override.reasoning_field.clone();
                }
            }
        }
    }
}

fn compare_field(lines: &mut Vec<String>, name: &str, runtime: &str, session: &str) {
    if runtime != session {
        lines.push(format!("{name}: runtime={runtime} session={session}"));
    }
}

fn compare_optional_field(
    lines: &mut Vec<String>,
    name: &str,
    runtime: Option<&str>,
    session: Option<&str>,
) {
    if runtime != session {
        lines.push(format!(
            "{name}: runtime={} session={}",
            runtime.unwrap_or("-"),
            session.unwrap_or("-")
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AppConfig;

    #[test]
    fn loads_provider_override_from_config() {
        let config: AppConfig = toml::from_str(
            r#"
[providers.compatible]
display_name = "Compatible Local"
api_key_env = "COMPATIBLE_TOKEN"
default_model = "custom/model"
default_summary_model = "custom/summary"
base_url = "http://localhost:4000/v1/chat/completions"

[providers.compatible.compat]
supports_reasoning_effort = false
supports_developer_role = false
reasoning_field = "reasoning_content"
"#,
        )
        .expect("parse config");

        let spec = Provider::Compatible.load_spec(&config).expect("load spec");
        assert_eq!(spec.display_name, "Compatible Local");
        assert_eq!(spec.api_key_env, "COMPATIBLE_TOKEN");
        assert_eq!(spec.default_model, "custom/model");
        assert_eq!(spec.default_summary_model, "custom/summary");
        assert_eq!(
            spec.resolved_base_url().as_deref(),
            Some("http://localhost:4000/v1/chat/completions")
        );

        match spec.backend {
            BackendSpec::ChatCompletions {
                default_base_url,
                compat,
                ..
            } => {
                assert_eq!(default_base_url, "http://localhost:11434/v1/chat/completions");
                assert!(!compat.supports_reasoning_effort);
                assert!(!compat.supports_developer_role);
                assert_eq!(compat.reasoning_field.as_deref(), Some("reasoning_content"));
            }
            other => panic!("unexpected backend: {other:?}"),
        }
    }

    #[test]
    fn detects_session_mismatch_lines() {
        let spec = Provider::Compatible
            .load_spec(&AppConfig::default())
            .expect("load default spec");
        let mut session = spec.to_session_info();
        session.model = "other/model".to_string();
        session.resolved_base_url = Some("http://old.example/v1/chat/completions".to_string());

        let lines = spec.mismatch_lines(&session);
        assert!(lines.iter().any(|line| line.contains("model")));
        assert!(lines.iter().any(|line| line.contains("resolved_base_url")));
    }
}

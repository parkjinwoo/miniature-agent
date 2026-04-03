mod app;
mod args;
mod bootstrap;
mod compaction;
mod config;
mod interactive;
mod non_interactive;
mod paths;
mod provider_registry;
mod runtime;
mod session_ui;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    app::run(std::env::args().collect()).await
}

#[cfg(test)]
mod tests {
    use super::session_ui::{
        format_session_summary_line, mismatch_resolution_options, mismatch_resolution_prompt,
        mismatch_summary_inline, mismatch_summary_lines, prioritize_matching_sessions,
        session_matches_provider,
    };
    use agent_model::{
        Backend, Capability, LlmMessage, LlmRole, MessagePart, ModelEvent, ModelEventStream,
        ModelRequest, StopReason, TextPart,
    };
    use agent_session::SessionStore;
    use async_trait::async_trait;
    use futures::stream;
    use crate::args::{
        parse_model, parse_prompt, parse_provider,
    };
    use crate::app::infer_default_provider;
    use crate::compaction::compact_session;
    use crate::config::AppConfig;
    use crate::provider_registry::{BackendSpec, Provider, ProviderSpec};
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct FakeSummaryBackend {
        events: Arc<Mutex<Vec<ModelEvent>>>,
    }

    #[async_trait]
    impl Backend for FakeSummaryBackend {
        fn name(&self) -> &'static str {
            "fake-summary"
        }

        fn supports(&self, _capability: Capability) -> bool {
            true
        }

        async fn stream(
            &self,
            _request: ModelRequest,
        ) -> anyhow::Result<ModelEventStream> {
            let events = self.events.lock().unwrap().clone();
            Ok(Box::pin(stream::iter(events.into_iter().map(Ok))))
        }
    }

    fn openai_spec() -> ProviderSpec {
        ProviderSpec {
            display_name: "OpenAI",
            api_key_env: "MINIATURE_AGENT_OPENAI_API_KEY".to_string(),
            requires_api_key: true,
            default_model: "gpt-5".to_string(),
            default_summary_model: "gpt-5".to_string(),
            base_url: None,
            base_url_env: Some("MINIATURE_AGENT_OPENAI_BASE_URL".to_string()),
            backend: BackendSpec::OpenAiResponses,
        }
    }

    fn temp_session_dir() -> PathBuf {
        let unique = uuid::Uuid::new_v4();
        let path = std::env::temp_dir().join(format!("miniature-agent-cli-{unique}"));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn summary(
        created_at: &str,
        provider_display_name: Option<&str>,
        provider_backend: Option<&str>,
        provider_model: Option<&str>,
    ) -> agent_session::SessionSummary {
        agent_session::SessionSummary {
            path: PathBuf::from(format!("/tmp/{created_at}.jsonl")),
            id: created_at.to_string(),
            cwd: "/workspace".to_string(),
            created_at: created_at.to_string(),
            parent_session: None,
            provider_display_name: provider_display_name.map(ToString::to_string),
            provider_backend: provider_backend.map(ToString::to_string),
            provider_model: provider_model.map(ToString::to_string),
            message_count: 3,
            summary_count: 1,
        }
    }

    fn text_message(role: LlmRole, text: &str) -> agent_core::AgentMessage {
        let message = LlmMessage {
            role: role.clone(),
            parts: vec![MessagePart::Text(TextPart {
                text: text.to_string(),
            })],
        };
        match role {
            LlmRole::User => agent_core::AgentMessage::User(message),
            LlmRole::Assistant => agent_core::AgentMessage::Assistant(message),
            LlmRole::Tool | LlmRole::System => agent_core::AgentMessage::ToolResult(message),
        }
    }

    #[test]
    fn parse_helpers_read_cli_flags() {
        let args = vec![
            "agent-cli".to_string(),
            "--provider".to_string(),
            "compatible".to_string(),
            "--model=test-model".to_string(),
        ];

        assert_eq!(parse_provider(&args), Some(Provider::Compatible));
        assert_eq!(parse_model(&args).as_deref(), Some("test-model"));
        assert_eq!(parse_prompt(&args), None);
    }

    #[test]
    fn compatible_provider_does_not_require_api_key() {
        let spec = Provider::Compatible
            .load_spec(&AppConfig::default())
            .unwrap();
        assert!(!spec.requires_api_key);
    }

    #[test]
    fn parse_prompt_reads_cli_flag() {
        let args = vec![
            "agent-cli".to_string(),
            "--provider".to_string(),
            "openai".to_string(),
            "--prompt".to_string(),
            "hello".to_string(),
        ];
        assert_eq!(parse_prompt(&args).as_deref(), Some("hello"));

        let args = vec!["agent-cli".to_string(), "--prompt=hi".to_string()];
        assert_eq!(parse_prompt(&args).as_deref(), Some("hi"));
    }

    #[test]
    fn inferred_default_provider_prefers_single_configured_provider() {
        let config: AppConfig = toml::from_str(
            r#"
[providers.compatible]
base_url = "http://localhost:11434/v1/chat/completions"
default_model = "local-model"
"#,
        )
        .expect("parse config");

        assert_eq!(infer_default_provider(&config), Provider::Compatible);
    }

    #[test]
    fn inferred_default_provider_prefers_single_provider_env() {
        unsafe {
            std::env::set_var(
                "MINIATURE_AGENT_COMPATIBLE_API_KEY",
                "compatible-token",
            )
        };
        assert_eq!(infer_default_provider(&AppConfig::default()), Provider::Compatible);
        unsafe { std::env::remove_var("MINIATURE_AGENT_COMPATIBLE_API_KEY") };
    }

    #[test]
    fn session_matching_and_mismatch_rendering_are_consistent() {
        let spec = openai_spec();
        let matching = summary(
            "2026-01-01T00:00:00Z",
            Some("OpenAI"),
            Some("openai-responses"),
            Some("gpt-5"),
        );
        let mismatched = summary(
            "2026-01-02T00:00:00Z",
            Some("Compatible"),
            Some("chat-completions"),
            Some("your-model"),
        );

        assert!(session_matches_provider(&matching, &spec));
        assert!(!session_matches_provider(&mismatched, &spec));

        let rendered_match = format_session_summary_line(&matching, Some(&spec));
        let rendered_mismatch = format_session_summary_line(&mismatched, Some(&spec));
        assert!(rendered_match.starts_with("= "));
        assert!(rendered_mismatch.starts_with("! "));
        assert!(rendered_mismatch.contains("diff="));

        let mismatch_lines = mismatch_summary_lines(&mismatched, &spec);
        assert!(mismatch_lines.iter().any(|line| line.contains("backend:")));
        assert!(mismatch_lines.iter().any(|line| line.contains("model:")));
    }

    #[test]
    fn prioritize_matching_sessions_places_current_provider_first() {
        let spec = openai_spec();
        let sessions = vec![
            summary(
                "2026-01-01T00:00:00Z",
                Some("Compatible"),
                Some("chat-completions"),
                Some("your-model"),
            ),
            summary(
                "2026-01-02T00:00:00Z",
                Some("OpenAI"),
                Some("openai-responses"),
                Some("gpt-5"),
            ),
            summary("2026-01-03T00:00:00Z", None, None, None),
        ];

        let prioritized = prioritize_matching_sessions(sessions, &spec);
        assert_eq!(
            prioritized[0].provider_display_name.as_deref(),
            Some("OpenAI")
        );
        assert_eq!(
            prioritized[1].provider_display_name.as_deref(),
            Some("Compatible")
        );
    }

    #[test]
    fn mismatch_resolution_prompt_and_options_are_actionable() {
        let spec = openai_spec();
        let mismatched = summary(
            "2026-01-02T00:00:00Z",
            Some("Compatible"),
            Some("chat-completions"),
            Some("your-model"),
        );

        let prompt = mismatch_resolution_prompt(&mismatched, &spec);
        let options = mismatch_resolution_options(&spec);

        assert!(prompt.contains("runtime=OpenAI/gpt-5"));
        assert!(prompt.contains("session=Compatible/your-model"));
        assert!(prompt.contains("recommended: use /fork"));
        assert_eq!(options.len(), 3);
        assert!(options[0].contains("Fork with current provider"));
        assert!(options[0].contains("[recommended]"));
    }

    #[test]
    fn mismatch_resolution_inline_summary_is_compact() {
        let spec = openai_spec();
        let mismatched = summary(
            "2026-01-02T00:00:00Z",
            Some("Compatible"),
            Some("chat-completions"),
            Some("your-model"),
        );

        let inline = mismatch_summary_inline(&mismatched, &spec);
        assert!(inline.contains("backend openai-responses ≠ chat-completions"));
        assert!(inline.contains("model gpt-5 ≠ your-model"));
    }

    #[tokio::test]
    async fn compact_session_uses_model_summary_when_available() {
        let session_dir = temp_session_dir();
        let mut session = SessionStore::create(&session_dir, "/workspace", None, None).unwrap();
        session
            .append_run(&agent_core::AgentRunResult {
                new_messages: vec![
                    text_message(LlmRole::User, "u1"),
                    text_message(LlmRole::Assistant, "a1"),
                    text_message(LlmRole::User, "u2"),
                    text_message(LlmRole::Assistant, "a2"),
                ],
                ..agent_core::AgentRunResult::default()
            })
            .unwrap();

        let backend = FakeSummaryBackend {
            events: Arc::new(Mutex::new(vec![
                ModelEvent::TextDelta("model summary".to_string()),
                ModelEvent::Completed {
                    stop_reason: StopReason::EndTurn,
                },
            ])),
        };

        compact_session(&mut session, 2, Some("key".to_string()), &backend, "summary-model")
            .await
            .unwrap()
            .unwrap();

        let restored = session.messages();
        let first_text = match &restored[0].as_llm_message().parts[0] {
            MessagePart::Text(part) => part.text.clone(),
            _ => panic!("expected summary text"),
        };
        assert!(first_text.contains("model summary"));

        let _ = std::fs::remove_dir_all(session_dir);
    }

    #[tokio::test]
    async fn compact_session_falls_back_when_summary_backend_errors() {
        let session_dir = temp_session_dir();
        let mut session = SessionStore::create(&session_dir, "/workspace", None, None).unwrap();
        session
            .append_run(&agent_core::AgentRunResult {
                new_messages: vec![
                    text_message(LlmRole::User, "u1"),
                    text_message(LlmRole::Assistant, "a1"),
                    text_message(LlmRole::User, "u2"),
                    text_message(LlmRole::Assistant, "a2"),
                ],
                ..agent_core::AgentRunResult::default()
            })
            .unwrap();

        let backend = FakeSummaryBackend {
            events: Arc::new(Mutex::new(vec![ModelEvent::Error("boom".to_string())])),
        };

        compact_session(&mut session, 2, Some("key".to_string()), &backend, "summary-model")
            .await
            .unwrap()
            .unwrap();

        let restored = session.messages();
        let first_text = match &restored[0].as_llm_message().parts[0] {
            MessagePart::Text(part) => part.text.clone(),
            _ => panic!("expected summary text"),
        };
        assert!(first_text.contains("- user: u1"));
        assert!(first_text.contains("- assistant: a1"));

        let _ = std::fs::remove_dir_all(session_dir);
    }

    #[tokio::test]
    async fn compact_session_falls_back_when_summary_backend_returns_empty_text() {
        let session_dir = temp_session_dir();
        let mut session = SessionStore::create(&session_dir, "/workspace", None, None).unwrap();
        session
            .append_run(&agent_core::AgentRunResult {
                new_messages: vec![
                    text_message(LlmRole::User, "u1"),
                    text_message(LlmRole::Assistant, "a1"),
                    text_message(LlmRole::User, "u2"),
                    text_message(LlmRole::Assistant, "a2"),
                ],
                ..agent_core::AgentRunResult::default()
            })
            .unwrap();

        let backend = FakeSummaryBackend {
            events: Arc::new(Mutex::new(vec![ModelEvent::Completed {
                stop_reason: StopReason::EndTurn,
            }])),
        };

        compact_session(&mut session, 2, Some("key".to_string()), &backend, "summary-model")
            .await
            .unwrap()
            .unwrap();

        let restored = session.messages();
        let first_text = match &restored[0].as_llm_message().parts[0] {
            MessagePart::Text(part) => part.text.clone(),
            _ => panic!("expected summary text"),
        };
        assert!(first_text.contains("- user: u1"));
        assert!(first_text.contains("- assistant: a1"));

        let _ = std::fs::remove_dir_all(session_dir);
    }
}

use anyhow::Context;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use agent_model::{
    Backend, LlmMessage, LlmRole, MessagePart, ModelEvent, ModelRequest, StopReason, ToolResultPart,
    Usage,
};
use agent_tools::{ToolCall, ToolRegistry};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentMessage {
    User(LlmMessage),
    Assistant(LlmMessage),
    ToolResult(LlmMessage),
}

impl AgentMessage {
    pub fn into_llm_message(self) -> LlmMessage {
        match self {
            AgentMessage::User(message)
            | AgentMessage::Assistant(message)
            | AgentMessage::ToolResult(message) => message,
        }
    }

    pub fn as_llm_message(&self) -> &LlmMessage {
        match self {
            AgentMessage::User(message)
            | AgentMessage::Assistant(message)
            | AgentMessage::ToolResult(message) => message,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentEvent {
    AgentStart,
    TurnStart,
    MessageStart { role: LlmRole },
    TextDelta(String),
    ToolCallStart { id: String, name: String },
    ToolCallArgsDelta { id: String, delta: String },
    ToolCallEnd { id: String },
    Usage(Usage),
    MessageEnd {
        message: LlmMessage,
        stop_reason: StopReason,
    },
    ToolResultReady { message: LlmMessage },
    TurnEnd {
        stop_reason: StopReason,
    },
    AgentEnd,
}

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub system: String,
    pub model: String,
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct AgentState {
    pub messages: Vec<AgentMessage>,
}

#[derive(Debug, Default)]
pub struct AgentRunResult {
    pub new_messages: Vec<AgentMessage>,
    pub events: Vec<AgentEvent>,
    pub stop_reason: Option<StopReason>,
}

pub struct Agent<B> {
    backend: B,
    config: AgentConfig,
    state: AgentState,
    tools: ToolRegistry,
}

impl<B> Agent<B>
where
    B: Backend,
{
    pub fn new(backend: B, config: AgentConfig, tools: ToolRegistry) -> Self {
        Self {
            backend,
            config,
            state: AgentState::default(),
            tools,
        }
    }

    pub fn state(&self) -> &AgentState {
        &self.state
    }

    pub fn set_state(&mut self, state: AgentState) {
        self.state = state;
    }

    pub fn into_state(self) -> AgentState {
        self.state
    }

    pub async fn prompt(
        &mut self,
        user_message: LlmMessage,
        api_key: secrecy::SecretString,
    ) -> anyhow::Result<AgentRunResult> {
        self.prompt_with(user_message, api_key, ignore_event).await
    }

    pub async fn prompt_with<F>(
        &mut self,
        user_message: LlmMessage,
        api_key: secrecy::SecretString,
        mut on_event: F,
    ) -> anyhow::Result<AgentRunResult>
    where
        F: FnMut(&AgentEvent) + Send,
    {
        let user_message = match user_message.role {
            LlmRole::User => user_message,
            role => {
                anyhow::bail!("prompt expects a user message, got {:?}", role);
            }
        };

        self.state
            .messages
            .push(AgentMessage::User(user_message.clone()));

        let mut result = AgentRunResult::default();
        emit_event(&mut result, AgentEvent::AgentStart, &mut on_event);
        emit_event(
            &mut result,
            AgentEvent::MessageStart {
                role: LlmRole::User,
            },
            &mut on_event,
        );
        emit_event(
            &mut result,
            AgentEvent::MessageEnd {
            message: user_message.clone(),
            stop_reason: StopReason::EndTurn,
        },
            &mut on_event,
        );

        result.new_messages.push(AgentMessage::User(user_message));

        loop {
            let continuation = self.run_turn(api_key.clone(), &mut on_event).await?;
            let should_continue = matches!(continuation.stop_reason, Some(StopReason::ToolCalls));
            result.stop_reason = continuation.stop_reason.clone();
            result.events.extend(continuation.events);
            result.new_messages.extend(continuation.new_messages);

            if !should_continue {
                break;
            }
        }

        Ok(result)
    }

    async fn run_turn(
        &mut self,
        api_key: secrecy::SecretString,
        on_event: &mut impl FnMut(&AgentEvent),
    ) -> anyhow::Result<AgentRunResult> {
        let request = ModelRequest {
            system: self.config.system.clone(),
            messages: self
                .state
                .messages
                .iter()
                .map(AgentMessage::as_llm_message)
                .cloned()
                .collect(),
            tools: self.tools.specs(),
            model: self.config.model.clone(),
            api_key,
            temperature: self.config.temperature,
        };

        let mut stream = self.backend.stream(request).await?;
        let mut assistant_parts = Vec::new();
        let mut result = AgentRunResult::default();
        emit_event(&mut result, AgentEvent::TurnStart, on_event);
        emit_event(
            &mut result,
            AgentEvent::MessageStart {
                role: LlmRole::Assistant,
            },
            on_event,
        );
        let mut stop_reason = StopReason::EndTurn;
        let mut tool_args_by_call_id: HashMap<String, String> = HashMap::new();

        while let Some(event) = stream.next().await {
            match event.context("backend stream failed")? {
                ModelEvent::TextDelta(delta) => {
                    push_text_delta(&mut assistant_parts, &delta);
                    emit_event(&mut result, AgentEvent::TextDelta(delta), on_event);
                }
                ModelEvent::ToolCallStart { id, name } => {
                    assistant_parts.push(MessagePart::ToolCall(agent_model::ToolCallPart {
                        id: id.clone(),
                        call_id: id.clone(),
                        name: name.clone(),
                        arguments_json: String::new(),
                    }));
                    tool_args_by_call_id.entry(id.clone()).or_default();
                    emit_event(&mut result, AgentEvent::ToolCallStart { id, name }, on_event);
                }
                ModelEvent::ToolCallArgsDelta { id, delta } => {
                    tool_args_by_call_id.entry(id.clone()).or_default().push_str(&delta);
                    emit_event(
                        &mut result,
                        AgentEvent::ToolCallArgsDelta { id, delta },
                        on_event,
                    );
                }
                ModelEvent::ToolCallEnd { id } => {
                    if let Some(final_args) = tool_args_by_call_id.get(&id) {
                        if let Some(MessagePart::ToolCall(tool_call)) = assistant_parts
                            .iter_mut()
                            .rev()
                            .find(|part| matches!(part, MessagePart::ToolCall(call) if call.call_id == id))
                        {
                            tool_call.arguments_json = final_args.clone();
                        }
                    }
                    emit_event(&mut result, AgentEvent::ToolCallEnd { id }, on_event);
                }
                ModelEvent::Usage(usage) => {
                    emit_event(&mut result, AgentEvent::Usage(usage), on_event);
                }
                ModelEvent::Completed { stop_reason: reason } => {
                    stop_reason = reason;
                }
                ModelEvent::Error(message) => {
                    stop_reason = StopReason::Error;
                    push_text_delta(&mut assistant_parts, &format!("Backend error: {message}"));
                }
            }
        }

        let assistant_message = LlmMessage {
            role: LlmRole::Assistant,
            parts: assistant_parts,
        };

        self.state
            .messages
            .push(AgentMessage::Assistant(assistant_message.clone()));

        emit_event(
            &mut result,
            AgentEvent::MessageEnd {
                message: assistant_message.clone(),
                stop_reason: stop_reason.clone(),
            },
            on_event,
        );
        emit_event(
            &mut result,
            AgentEvent::TurnEnd {
                stop_reason: stop_reason.clone(),
            },
            on_event,
        );
        let mut new_messages = vec![AgentMessage::Assistant(assistant_message.clone())];

        if matches!(stop_reason, StopReason::ToolCalls) {
            for part in &assistant_message.parts {
                if let MessagePart::ToolCall(tool_call) = part {
                    let call = ToolCall {
                        id: tool_call.call_id.clone(),
                        name: tool_call.name.clone(),
                        arguments_json: tool_call.arguments_json.clone(),
                    };
                    let execution = self.tools.execute(&call);
                    let (content, is_error) = match execution {
                        Ok(output) => (output.content, output.is_error),
                        Err(error) => (error.to_string(), true),
                    };

                    let tool_result_message = LlmMessage {
                        role: LlmRole::Tool,
                        parts: vec![MessagePart::ToolResult(ToolResultPart {
                            call_id: tool_call.call_id.clone(),
                            content,
                            is_error,
                        })],
                    };

                    self.state
                        .messages
                        .push(AgentMessage::ToolResult(tool_result_message.clone()));
                    emit_event(
                        &mut result,
                        AgentEvent::ToolResultReady {
                            message: tool_result_message.clone(),
                        },
                        on_event,
                    );
                    new_messages.push(AgentMessage::ToolResult(tool_result_message));
                }
            }
        }

        if !matches!(stop_reason, StopReason::ToolCalls) {
            emit_event(&mut result, AgentEvent::AgentEnd, on_event);
        }

        result.new_messages = new_messages;
        result.stop_reason = Some(stop_reason);
        Ok(result)
    }
}

fn emit_event(result: &mut AgentRunResult, event: AgentEvent, on_event: &mut impl FnMut(&AgentEvent)) {
    on_event(&event);
    result.events.push(event);
}

fn ignore_event(_: &AgentEvent) {}

fn push_text_delta(parts: &mut Vec<MessagePart>, delta: &str) {
    if let Some(MessagePart::Text(text)) = parts.last_mut() {
        text.text.push_str(delta);
    } else {
        parts.push(MessagePart::Text(agent_model::TextPart {
            text: delta.to_string(),
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_model::{Capability, ModelEventStream, StopReason, ToolSpec};
    use agent_tools::{Tool, ToolCall, ToolOutput};
    use async_trait::async_trait;
    use futures::stream;
    use secrecy::SecretString;
    use serde_json::json;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct FakeBackend {
        streams: Arc<Mutex<VecDeque<Vec<ModelEvent>>>>,
    }

    #[async_trait]
    impl Backend for FakeBackend {
        fn name(&self) -> &'static str {
            "fake"
        }

        fn supports(&self, _capability: Capability) -> bool {
            true
        }

        async fn stream(&self, _request: ModelRequest) -> anyhow::Result<ModelEventStream> {
            let events = self.streams.lock().unwrap().pop_front().unwrap_or_default();
            Ok(Box::pin(stream::iter(events.into_iter().map(Ok))))
        }
    }

    struct EchoTool;

    impl Tool for EchoTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: "echo".to_string(),
                description: "echo".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "value": { "type": "string" }
                    }
                }),
            }
        }

        fn run(&self, call: &ToolCall) -> anyhow::Result<ToolOutput> {
            let value = serde_json::from_str::<serde_json::Value>(&call.arguments_json)?
                .get("value")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string();
            Ok(ToolOutput {
                content: format!("echo:{value}"),
                is_error: false,
            })
        }
    }

    fn user_message(text: &str) -> LlmMessage {
        LlmMessage {
            role: LlmRole::User,
            parts: vec![MessagePart::Text(agent_model::TextPart {
                text: text.to_string(),
            })],
        }
    }

    #[tokio::test]
    async fn prompt_runs_tool_call_and_continues_to_final_assistant_message() {
        let backend = FakeBackend {
            streams: Arc::new(Mutex::new(VecDeque::from(vec![
                vec![
                    ModelEvent::ToolCallStart {
                        id: "call-1".to_string(),
                        name: "echo".to_string(),
                    },
                    ModelEvent::ToolCallArgsDelta {
                        id: "call-1".to_string(),
                        delta: r#"{"value":"hello"}"#.to_string(),
                    },
                    ModelEvent::ToolCallEnd {
                        id: "call-1".to_string(),
                    },
                    ModelEvent::Completed {
                        stop_reason: StopReason::ToolCalls,
                    },
                ],
                vec![
                    ModelEvent::TextDelta("done".to_string()),
                    ModelEvent::Completed {
                        stop_reason: StopReason::EndTurn,
                    },
                ],
            ]))),
        };

        let mut tools = ToolRegistry::new();
        tools.register(EchoTool);
        let mut agent = Agent::new(
            backend,
            AgentConfig {
                system: "system".to_string(),
                model: "test-model".to_string(),
                temperature: None,
            },
            tools,
        );

        let run = agent
            .prompt(user_message("hi"), SecretString::new("key".to_string().into_boxed_str()))
            .await
            .unwrap();

        assert_eq!(run.stop_reason, Some(StopReason::EndTurn));
        assert_eq!(run.new_messages.len(), 4);
        assert!(matches!(run.new_messages[0], AgentMessage::User(_)));
        assert!(matches!(run.new_messages[1], AgentMessage::Assistant(_)));
        assert!(matches!(run.new_messages[2], AgentMessage::ToolResult(_)));
        assert!(matches!(run.new_messages[3], AgentMessage::Assistant(_)));

        let state = agent.state();
        assert_eq!(state.messages.len(), 4);
        let tool_result = state.messages[2].as_llm_message();
        assert!(matches!(&tool_result.parts[0], MessagePart::ToolResult(part) if part.content == "echo:hello"));
        let final_message = state.messages[3].as_llm_message();
        assert!(matches!(&final_message.parts[0], MessagePart::Text(part) if part.text == "done"));
    }

    #[tokio::test]
    async fn prompt_records_backend_error_and_stops() {
        let backend = FakeBackend {
            streams: Arc::new(Mutex::new(VecDeque::from(vec![vec![
                ModelEvent::Error("backend failed".to_string()),
                ModelEvent::Completed {
                    stop_reason: StopReason::Error,
                },
            ]]))),
        };

        let agent_events = Arc::new(Mutex::new(Vec::<AgentEvent>::new()));
        let captured_events = Arc::clone(&agent_events);
        let mut agent = Agent::new(
            backend,
            AgentConfig {
                system: "system".to_string(),
                model: "test-model".to_string(),
                temperature: None,
            },
            ToolRegistry::new(),
        );

        let run = agent
            .prompt_with(
                user_message("hi"),
                SecretString::new("key".to_string().into_boxed_str()),
                move |event| captured_events.lock().unwrap().push(event.clone()),
            )
            .await
            .unwrap();

        assert_eq!(run.stop_reason, Some(StopReason::Error));
        assert_eq!(run.new_messages.len(), 2);
        let assistant = run.new_messages[1].as_llm_message();
        assert!(matches!(
            &assistant.parts[0],
            MessagePart::Text(part) if part.text.contains("Backend error: backend failed")
        ));

        let events = agent_events.lock().unwrap();
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::TurnEnd { stop_reason } if stop_reason == &StopReason::Error
        )));
        assert!(events.iter().any(|event| matches!(event, AgentEvent::AgentEnd)));
    }
}

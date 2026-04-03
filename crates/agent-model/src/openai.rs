use std::collections::{HashMap, HashSet};

use anyhow::Context;
use async_stream::try_stream;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use secrecy::ExposeSecret;

use crate::{Backend, Capability, LlmRole, MessagePart, ModelEvent, ModelEventStream, ModelRequest, StopReason};

const RESPONSES_API_URL: &str = "https://api.openai.com/v1/responses";

#[derive(Debug, Clone)]
pub struct OpenAiBackend {
    client: Client,
    base_url: String,
}

impl Default for OpenAiBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAiBackend {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: RESPONSES_API_URL.to_string(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn build_request_body(&self, request: &ModelRequest) -> serde_json::Value {
        let input = request
            .messages
            .iter()
            .filter_map(|message| match message.role {
                LlmRole::Tool => message.parts.iter().find_map(|part| match part {
                    MessagePart::ToolResult(result) => Some(serde_json::json!({
                        "type": "function_call_output",
                        "call_id": result.call_id,
                        "output": result.content,
                    })),
                    _ => None,
                }),
                LlmRole::System | LlmRole::User | LlmRole::Assistant => {
                    let role = match message.role {
                        LlmRole::System => "system",
                        LlmRole::User => "user",
                        LlmRole::Assistant => "assistant",
                        LlmRole::Tool => unreachable!(),
                    };

                    let content = message
                        .parts
                        .iter()
                        .filter_map(|part| match part {
                            MessagePart::Text(text) => Some(serde_json::json!({
                                "type": if matches!(message.role, LlmRole::Assistant) {
                                    "output_text"
                                } else {
                                    "input_text"
                                },
                                "text": text.text,
                            })),
                            MessagePart::ToolCall(_) | MessagePart::ToolResult(_) => None,
                        })
                        .collect::<Vec<_>>();

                    Some(serde_json::json!({
                        "role": role,
                        "content": content,
                    }))
                }
            })
            .collect::<Vec<_>>();

        let tools = request
            .tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                })
            })
            .collect::<Vec<_>>();

        let mut body = serde_json::json!({
            "model": request.model,
            "input": input,
            "tools": tools,
            "stream": true,
        });

        if !request.system.is_empty() {
            body["instructions"] = serde_json::Value::String(request.system.clone());
        }

        if let Some(temperature) = request.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }

        body
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn client(&self) -> &Client {
        &self.client
    }
}

#[async_trait]
impl Backend for OpenAiBackend {
    fn name(&self) -> &'static str {
        "openai"
    }

    fn supports(&self, capability: Capability) -> bool {
        matches!(
            capability,
            Capability::StreamingText | Capability::ToolCalling | Capability::Reasoning
        )
    }

    async fn stream(&self, request: ModelRequest) -> anyhow::Result<ModelEventStream> {
        let body = self.build_request_body(&request);
        let mut request_builder = self
            .client()
            .post(self.base_url())
            .header("content-type", "application/json");
        if !request.api_key.expose_secret().trim().is_empty() {
            request_builder = request_builder.bearer_auth(request.api_key.expose_secret());
        }
        let response = request_builder
            .json(&body)
            .send()
            .await
            .with_context(|| format!("failed to send request to {}", self.base_url()))?
            .error_for_status()
            .with_context(|| format!("OpenAI request failed for {}", self.base_url()))?;

        let stream = try_stream! {
            let mut buffer = String::new();
            let mut state = OpenAiStreamState::default();
            let mut bytes = response.bytes_stream();

            while let Some(chunk) = bytes.next().await {
                let chunk = chunk.context("failed to read OpenAI response chunk")?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(frame) = next_sse_frame(&mut buffer) {
                    if let Some(event) = parse_sse_data(&frame)? {
                        for translated in state.translate(event) {
                            yield translated;
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

#[derive(Debug)]
enum OpenAiSseEvent {
    OutputTextDelta { delta: String },
    OutputItemAdded { item_id: String, call_id: String, name: String },
    FunctionCallArgumentsDelta { item_id: String, delta: String },
    FunctionCallArgumentsDone { item_id: String, arguments: String },
    OutputItemDone { item_id: String, call_id: Option<String> },
    Completed,
    Failed { message: String },
    Error { message: String },
    Ignored,
}

#[derive(Default)]
struct OpenAiStreamState {
    saw_tool_call: bool,
    tool_call_by_item_id: HashMap<String, String>,
    saw_args_delta_for_item_id: HashSet<String>,
}

impl OpenAiStreamState {
    fn translate(&mut self, event: OpenAiSseEvent) -> Vec<ModelEvent> {
        match event {
            OpenAiSseEvent::OutputTextDelta { delta } => vec![ModelEvent::TextDelta(delta)],
            OpenAiSseEvent::OutputItemAdded { item_id, call_id, name } => {
                if call_id.is_empty() {
                    Vec::new()
                } else {
                    self.saw_tool_call = true;
                    self.tool_call_by_item_id.insert(item_id, call_id.clone());
                    vec![ModelEvent::ToolCallStart { id: call_id, name }]
                }
            }
            OpenAiSseEvent::FunctionCallArgumentsDelta { item_id, delta } => {
                self.saw_args_delta_for_item_id.insert(item_id.clone());
                self.tool_call_by_item_id
                    .get(&item_id)
                    .map(|call_id| {
                        vec![ModelEvent::ToolCallArgsDelta {
                            id: call_id.clone(),
                            delta,
                        }]
                    })
                    .unwrap_or_default()
            }
            OpenAiSseEvent::FunctionCallArgumentsDone { item_id, arguments } => {
                if self.saw_args_delta_for_item_id.contains(&item_id) {
                    Vec::new()
                } else {
                    self.tool_call_by_item_id
                        .get(&item_id)
                        .map(|call_id| {
                            vec![ModelEvent::ToolCallArgsDelta {
                                id: call_id.clone(),
                                delta: arguments,
                            }]
                        })
                        .unwrap_or_default()
                }
            }
            OpenAiSseEvent::OutputItemDone { item_id, call_id } => {
                self.saw_args_delta_for_item_id.remove(&item_id);
                let mapped_call_id = self.tool_call_by_item_id.remove(&item_id);
                let resolved = call_id.or(mapped_call_id).unwrap_or(item_id);
                vec![ModelEvent::ToolCallEnd { id: resolved }]
            }
            OpenAiSseEvent::Completed => {
                let stop_reason = if self.saw_tool_call {
                    StopReason::ToolCalls
                } else {
                    StopReason::EndTurn
                };
                vec![ModelEvent::Completed { stop_reason }]
            }
            OpenAiSseEvent::Failed { message } | OpenAiSseEvent::Error { message } => {
                vec![ModelEvent::Error(message)]
            }
            OpenAiSseEvent::Ignored => Vec::new(),
        }
    }
}

fn next_sse_frame(buffer: &mut String) -> Option<String> {
    if let Some(index) = buffer.find("\r\n\r\n") {
        let frame = buffer[..index].to_string();
        buffer.drain(..index + 4);
        return Some(frame);
    }

    if let Some(index) = buffer.find("\n\n") {
        let frame = buffer[..index].to_string();
        buffer.drain(..index + 2);
        return Some(frame);
    }

    None
}

fn parse_sse_data(frame: &str) -> anyhow::Result<Option<OpenAiSseEvent>> {
    let mut data_lines = Vec::new();

    for line in frame.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.trim_start());
        }
    }

    if data_lines.is_empty() {
        return Ok(None);
    }

    let data = data_lines.join("\n");
    if data == "[DONE]" {
        return Ok(None);
    }

    let value: serde_json::Value = serde_json::from_str(&data)
        .with_context(|| format!("failed to parse SSE payload: {data}"))?;
    let event_type = value
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();

    let event = match event_type {
        "response.output_text.delta" => OpenAiSseEvent::OutputTextDelta {
            delta: value
                .get("delta")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
        },
        "response.output_item.added" => {
            let item = value.get("item").cloned().unwrap_or_default();
            if item.get("type").and_then(serde_json::Value::as_str) == Some("function_call") {
                OpenAiSseEvent::OutputItemAdded {
                    item_id: item
                        .get("id")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    call_id: item
                        .get("call_id")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    name: item
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                }
            } else {
                OpenAiSseEvent::Ignored
            }
        }
        "response.function_call_arguments.delta" => OpenAiSseEvent::FunctionCallArgumentsDelta {
            item_id: value
                .get("item_id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
            delta: value
                .get("delta")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
        },
        "response.function_call_arguments.done" => OpenAiSseEvent::FunctionCallArgumentsDone {
            item_id: value
                .get("item_id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
            arguments: value
                .get("arguments")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
        },
        "response.output_item.done" => {
            let item = value.get("item").cloned().unwrap_or_default();
            if item.get("type").and_then(serde_json::Value::as_str) == Some("function_call") {
                OpenAiSseEvent::OutputItemDone {
                    item_id: item
                        .get("id")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    call_id: item
                        .get("call_id")
                        .and_then(serde_json::Value::as_str)
                        .map(ToString::to_string),
                }
            } else {
                OpenAiSseEvent::Ignored
            }
        }
        "response.completed" => OpenAiSseEvent::Completed,
        "response.failed" => OpenAiSseEvent::Failed {
            message: value
                .get("response")
                .and_then(|response| response.get("error"))
                .and_then(|error| error.get("message"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("response.failed")
                .to_string(),
        },
        "error" => OpenAiSseEvent::Error {
            message: value
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("error")
                .to_string(),
        },
        _ => OpenAiSseEvent::Ignored,
    };

    Ok(Some(event))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LlmMessage, TextPart, ToolResultPart};
    use secrecy::SecretString;

    fn request_with_messages(messages: Vec<LlmMessage>) -> ModelRequest {
        ModelRequest {
            system: "system".to_string(),
            messages,
            tools: Vec::new(),
            model: "gpt-5".to_string(),
            api_key: SecretString::new("key".to_string().into_boxed_str()),
            temperature: Some(0.2),
        }
    }

    #[test]
    fn build_request_body_maps_tool_results_to_function_call_output() {
        let backend = OpenAiBackend::new();
        let body = backend.build_request_body(&request_with_messages(vec![
            LlmMessage {
                role: LlmRole::User,
                parts: vec![MessagePart::Text(TextPart {
                    text: "hello".to_string(),
                })],
            },
            LlmMessage {
                role: LlmRole::Tool,
                parts: vec![MessagePart::ToolResult(ToolResultPart {
                    call_id: "call-1".to_string(),
                    content: "tool output".to_string(),
                    is_error: false,
                })],
            },
        ]));

        let input = body["input"].as_array().unwrap();
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[1]["type"], "function_call_output");
        assert_eq!(input[1]["call_id"], "call-1");
        assert_eq!(input[1]["output"], "tool output");
        assert_eq!(body["instructions"], "system");
        let temperature = body["temperature"].as_f64().unwrap();
        assert!((temperature - 0.2).abs() < 0.000_001);
    }

    #[test]
    fn parses_output_item_and_argument_events() {
        let added = parse_sse_data(
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"id\":\"item-1\",\"call_id\":\"call-1\",\"name\":\"read\"}}\n\n",
        )
        .unwrap()
        .unwrap();
        assert!(matches!(
            added,
            OpenAiSseEvent::OutputItemAdded { item_id, call_id, name }
            if item_id == "item-1" && call_id == "call-1" && name == "read"
        ));

        let delta = parse_sse_data(
            "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"item-1\",\"delta\":\"{\\\"path\\\":\"}\n\n",
        )
        .unwrap()
        .unwrap();
        assert!(matches!(
            delta,
            OpenAiSseEvent::FunctionCallArgumentsDelta { item_id, delta }
            if item_id == "item-1" && delta == "{\"path\":"
        ));

        let done = parse_sse_data(
            "data: {\"type\":\"response.function_call_arguments.done\",\"item_id\":\"item-1\",\"arguments\":\"{\\\"path\\\":\\\"Cargo.toml\\\"}\"}\n\n",
        )
        .unwrap()
        .unwrap();
        assert!(matches!(
            done,
            OpenAiSseEvent::FunctionCallArgumentsDone { item_id, arguments }
            if item_id == "item-1" && arguments == "{\"path\":\"Cargo.toml\"}"
        ));
    }

    #[test]
    fn parses_completed_and_error_events() {
        let completed = parse_sse_data("data: {\"type\":\"response.completed\"}\n\n")
            .unwrap()
            .unwrap();
        assert!(matches!(completed, OpenAiSseEvent::Completed));

        let failed = parse_sse_data(
            "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"bad request\"}}}\n\n",
        )
        .unwrap()
        .unwrap();
        assert!(matches!(
            failed,
            OpenAiSseEvent::Failed { message } if message == "bad request"
        ));
    }

    #[test]
    fn argument_done_does_not_duplicate_streamed_tool_args() {
        let mut state = OpenAiStreamState::default();
        let started = state.translate(OpenAiSseEvent::OutputItemAdded {
            item_id: "item-1".to_string(),
            call_id: "call-1".to_string(),
            name: "read".to_string(),
        });
        assert!(matches!(
            started.as_slice(),
            [ModelEvent::ToolCallStart { id, name }] if id == "call-1" && name == "read"
        ));

        let delta = state.translate(OpenAiSseEvent::FunctionCallArgumentsDelta {
            item_id: "item-1".to_string(),
            delta: "{\"path\":".to_string(),
        });
        assert!(matches!(
            delta.as_slice(),
            [ModelEvent::ToolCallArgsDelta { id, delta }] if id == "call-1" && delta == "{\"path\":"
        ));

        let done = state.translate(OpenAiSseEvent::FunctionCallArgumentsDone {
            item_id: "item-1".to_string(),
            arguments: "{\"path\":\"Cargo.toml\"}".to_string(),
        });
        assert!(done.is_empty());
    }

    #[test]
    fn argument_done_supplies_full_args_when_no_deltas_arrived() {
        let mut state = OpenAiStreamState::default();
        state.translate(OpenAiSseEvent::OutputItemAdded {
            item_id: "item-1".to_string(),
            call_id: "call-1".to_string(),
            name: "read".to_string(),
        });

        let done = state.translate(OpenAiSseEvent::FunctionCallArgumentsDone {
            item_id: "item-1".to_string(),
            arguments: "{\"path\":\"Cargo.toml\"}".to_string(),
        });
        assert!(matches!(
            done.as_slice(),
            [ModelEvent::ToolCallArgsDelta { id, delta }] if id == "call-1" && delta == "{\"path\":\"Cargo.toml\"}"
        ));
    }

    #[test]
    fn extracts_sse_frames() {
        let mut buffer = "data: one\n\ndata: two\n\n".to_string();
        assert_eq!(next_sse_frame(&mut buffer).as_deref(), Some("data: one"));
        assert_eq!(next_sse_frame(&mut buffer).as_deref(), Some("data: two"));
        assert!(next_sse_frame(&mut buffer).is_none());

        let mut buffer = "data: three\r\n\r\ndata: four\r\n\r\n".to_string();
        assert_eq!(next_sse_frame(&mut buffer).as_deref(), Some("data: three"));
        assert_eq!(next_sse_frame(&mut buffer).as_deref(), Some("data: four"));
        assert!(next_sse_frame(&mut buffer).is_none());
    }
}

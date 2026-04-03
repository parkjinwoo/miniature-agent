use std::collections::HashMap;

use crate::{
    Backend, Capability, LlmRole, MessagePart, ModelEvent, ModelEventStream, ModelRequest,
    StopReason,
};
use anyhow::Context;
use async_stream::try_stream;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use secrecy::ExposeSecret;

const MESSAGES_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 4096;

#[derive(Debug, Clone)]
pub struct AnthropicBackend {
    client: Client,
    base_url: String,
}

impl Default for AnthropicBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl AnthropicBackend {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: MESSAGES_API_URL.to_string(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn build_request_body(&self, request: &ModelRequest) -> serde_json::Value {
        let messages = request
            .messages
            .iter()
            .filter_map(|message| anthropic_message_from_llm(message))
            .collect::<Vec<_>>();

        let tools = request
            .tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                })
            })
            .collect::<Vec<_>>();

        let mut body = serde_json::json!({
            "model": request.model,
            "messages": messages,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "stream": true,
        });

        if !request.system.is_empty() {
            body["system"] = serde_json::Value::String(request.system.clone());
        }

        if !tools.is_empty() {
            body["tools"] = serde_json::Value::Array(tools);
            body["tool_choice"] = serde_json::json!({ "type": "auto" });
        }

        body
    }
}

#[async_trait]
impl Backend for AnthropicBackend {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    fn supports(&self, capability: Capability) -> bool {
        matches!(
            capability,
            Capability::StreamingText | Capability::ToolCalling | Capability::Reasoning
        )
    }

    async fn stream(&self, request: ModelRequest) -> anyhow::Result<ModelEventStream> {
        let body = self.build_request_body(&request);
        let response = self
            .client
            .post(&self.base_url)
            .header("content-type", "application/json")
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("x-api-key", request.api_key.expose_secret())
            .json(&body)
            .send()
            .await
            .with_context(|| format!("failed to send request to {}", self.base_url))?
            .error_for_status()
            .with_context(|| format!("Anthropic request failed for {}", self.base_url))?;

        let stream = try_stream! {
            let mut buffer = String::new();
            let mut bytes = response.bytes_stream();
            let mut tool_uses_by_index: HashMap<u64, ToolUseBlock> = HashMap::new();
            let mut stop_reason = StopReason::EndTurn;

            while let Some(chunk) = bytes.next().await {
                let chunk = chunk.context("failed to read Anthropic response chunk")?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(frame) = next_sse_frame(&mut buffer) {
                    if let Some(event) = parse_sse_data(&frame)? {
                        match event {
                            AnthropicSseEvent::TextDelta { text } => {
                                yield ModelEvent::TextDelta(text);
                            }
                            AnthropicSseEvent::ToolUseStart { index, id, name } => {
                                tool_uses_by_index.insert(index, ToolUseBlock {
                                    id: id.clone(),
                                });
                                yield ModelEvent::ToolCallStart { id, name };
                            }
                            AnthropicSseEvent::InputJsonDelta { index, partial_json } => {
                                if let Some(tool_use) = tool_uses_by_index.get(&index) {
                                    yield ModelEvent::ToolCallArgsDelta {
                                        id: tool_use.id.clone(),
                                        delta: partial_json,
                                    };
                                }
                            }
                            AnthropicSseEvent::ContentBlockStop { index } => {
                                if let Some(tool_use) = tool_uses_by_index.remove(&index) {
                                    yield ModelEvent::ToolCallEnd { id: tool_use.id };
                                }
                            }
                            AnthropicSseEvent::MessageDelta { new_stop_reason } => {
                                stop_reason = new_stop_reason;
                            }
                            AnthropicSseEvent::MessageStop => {
                                yield ModelEvent::Completed { stop_reason: stop_reason.clone() };
                            }
                            AnthropicSseEvent::Error { message } => {
                                yield ModelEvent::Error(message);
                            }
                            AnthropicSseEvent::Ignored => {}
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

#[derive(Debug)]
struct ToolUseBlock {
    id: String,
}

#[derive(Debug)]
enum AnthropicSseEvent {
    TextDelta { text: String },
    ToolUseStart { index: u64, id: String, name: String },
    InputJsonDelta { index: u64, partial_json: String },
    ContentBlockStop { index: u64 },
    MessageDelta { new_stop_reason: StopReason },
    MessageStop,
    Error { message: String },
    Ignored,
}

fn anthropic_message_from_llm(message: &crate::LlmMessage) -> Option<serde_json::Value> {
    match message.role {
        LlmRole::System => None,
        LlmRole::User => {
            let content = message
                .parts
                .iter()
                .filter_map(|part| match part {
                    MessagePart::Text(text) => Some(serde_json::json!({
                        "type": "text",
                        "text": text.text,
                    })),
                    MessagePart::ToolCall(_) | MessagePart::ToolResult(_) => None,
                })
                .collect::<Vec<_>>();
            Some(serde_json::json!({
                "role": "user",
                "content": content,
            }))
        }
        LlmRole::Assistant => {
            let content = message
                .parts
                .iter()
                .filter_map(|part| match part {
                    MessagePart::Text(text) => Some(serde_json::json!({
                        "type": "text",
                        "text": text.text,
                    })),
                    MessagePart::ToolCall(call) => Some(serde_json::json!({
                        "type": "tool_use",
                        "id": call.call_id,
                        "name": call.name,
                        "input": parse_json_object_or_empty(&call.arguments_json),
                    })),
                    MessagePart::ToolResult(_) => None,
                })
                .collect::<Vec<_>>();
            Some(serde_json::json!({
                "role": "assistant",
                "content": content,
            }))
        }
        LlmRole::Tool => {
            let mut tool_results = Vec::new();
            let mut trailing_text = Vec::new();

            for part in &message.parts {
                match part {
                    MessagePart::ToolResult(result) => {
                        tool_results.push(serde_json::json!({
                            "type": "tool_result",
                            "tool_use_id": result.call_id,
                            "content": result.content,
                            "is_error": result.is_error,
                        }));
                    }
                    MessagePart::Text(text) => {
                        trailing_text.push(serde_json::json!({
                            "type": "text",
                            "text": text.text,
                        }));
                    }
                    MessagePart::ToolCall(_) => {}
                }
            }

            tool_results.extend(trailing_text);
            Some(serde_json::json!({
                "role": "user",
                "content": tool_results,
            }))
        }
    }
}

fn parse_json_object_or_empty(input: &str) -> serde_json::Value {
    match serde_json::from_str::<serde_json::Value>(input) {
        Ok(value @ serde_json::Value::Object(_)) => value,
        _ => serde_json::json!({}),
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

fn parse_sse_data(frame: &str) -> anyhow::Result<Option<AnthropicSseEvent>> {
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
    let event = match value
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
    {
        "content_block_start" => {
            let index = value
                .get("index")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_default();
            let block = value.get("content_block").cloned().unwrap_or_default();
            if block.get("type").and_then(serde_json::Value::as_str) == Some("tool_use") {
                AnthropicSseEvent::ToolUseStart {
                    index,
                    id: block
                        .get("id")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    name: block
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                }
            } else {
                AnthropicSseEvent::Ignored
            }
        }
        "content_block_delta" => {
            let index = value
                .get("index")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_default();
            let delta = value.get("delta").cloned().unwrap_or_default();
            match delta
                .get("type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
            {
                "text_delta" => AnthropicSseEvent::TextDelta {
                    text: delta
                        .get("text")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                },
                "input_json_delta" => AnthropicSseEvent::InputJsonDelta {
                    index,
                    partial_json: delta
                        .get("partial_json")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                },
                _ => AnthropicSseEvent::Ignored,
            }
        }
        "content_block_stop" => AnthropicSseEvent::ContentBlockStop {
            index: value
                .get("index")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_default(),
        },
        "message_delta" => AnthropicSseEvent::MessageDelta {
            new_stop_reason: map_stop_reason(
                value
                    .get("delta")
                    .and_then(|delta| delta.get("stop_reason"))
                    .and_then(serde_json::Value::as_str),
            ),
        },
        "message_stop" => AnthropicSseEvent::MessageStop,
        "error" => AnthropicSseEvent::Error {
            message: value
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
        },
        "message_start" | "ping" => AnthropicSseEvent::Ignored,
        _ => AnthropicSseEvent::Ignored,
    };

    Ok(Some(event))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LlmMessage, ToolCallPart, ToolResultPart};
    use secrecy::SecretString;

    fn request_with_messages(messages: Vec<LlmMessage>) -> ModelRequest {
        ModelRequest {
            system: "system".to_string(),
            messages,
            tools: Vec::new(),
            model: "claude-sonnet".to_string(),
            api_key: SecretString::new("key".to_string().into_boxed_str()),
            temperature: None,
        }
    }

    #[test]
    fn build_request_body_maps_assistant_tool_use_and_tool_result() {
        let backend = AnthropicBackend::new();
        let body = backend.build_request_body(&request_with_messages(vec![
            LlmMessage {
                role: LlmRole::Assistant,
                parts: vec![MessagePart::ToolCall(ToolCallPart {
                    id: "item-1".to_string(),
                    call_id: "call-1".to_string(),
                    name: "read".to_string(),
                    arguments_json: r#"{"path":"Cargo.toml"}"#.to_string(),
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

        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["content"][0]["type"], "tool_use");
        assert_eq!(messages[0]["content"][0]["id"], "call-1");
        assert_eq!(messages[0]["content"][0]["input"]["path"], "Cargo.toml");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"][0]["type"], "tool_result");
        assert_eq!(messages[1]["content"][0]["tool_use_id"], "call-1");
        assert_eq!(messages[1]["content"][0]["content"], "tool output");
        assert_eq!(body["system"], "system");
    }

    #[test]
    fn parses_tool_use_and_text_delta_events() {
        let start = parse_sse_data(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call-1\",\"name\":\"read\"}}\n\n",
        )
        .unwrap()
        .unwrap();
        assert!(matches!(
            start,
            AnthropicSseEvent::ToolUseStart { index, id, name }
            if index == 0 && id == "call-1" && name == "read"
        ));

        let delta = parse_sse_data(
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\"}}\n\n",
        )
        .unwrap()
        .unwrap();
        assert!(matches!(
            delta,
            AnthropicSseEvent::InputJsonDelta { index, partial_json }
            if index == 0 && partial_json == "{\"path\":"
        ));

        let text = parse_sse_data(
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n",
        )
        .unwrap()
        .unwrap();
        assert!(matches!(
            text,
            AnthropicSseEvent::TextDelta { text } if text == "hello"
        ));
    }

    #[test]
    fn parses_stop_reason_and_errors() {
        let delta = parse_sse_data(
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n",
        )
        .unwrap()
        .unwrap();
        assert!(matches!(
            delta,
            AnthropicSseEvent::MessageDelta { new_stop_reason }
            if new_stop_reason == StopReason::ToolCalls
        ));

        let stop = parse_sse_data("data: {\"type\":\"message_stop\"}\n\n")
            .unwrap()
            .unwrap();
        assert!(matches!(stop, AnthropicSseEvent::MessageStop));

        let error = parse_sse_data(
            "data: {\"type\":\"error\",\"error\":{\"message\":\"bad request\"}}\n\n",
        )
        .unwrap()
        .unwrap();
        assert!(matches!(
            error,
            AnthropicSseEvent::Error { message } if message == "bad request"
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

    #[test]
    fn parse_json_object_or_empty_rejects_non_object() {
        assert_eq!(parse_json_object_or_empty(r#"{"a":1}"#)["a"], 1);
        assert_eq!(parse_json_object_or_empty(r#"[1,2,3]"#), serde_json::json!({}));
        assert_eq!(parse_json_object_or_empty("not json"), serde_json::json!({}));
    }
}

fn map_stop_reason(stop_reason: Option<&str>) -> StopReason {
    match stop_reason.unwrap_or_default() {
        "tool_use" => StopReason::ToolCalls,
        "max_tokens" => StopReason::MaxTokens,
        "pause_turn" => StopReason::Cancelled,
        "refusal" => StopReason::Error,
        _ => StopReason::EndTurn,
    }
}

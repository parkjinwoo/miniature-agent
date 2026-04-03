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

const DEFAULT_CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";

#[derive(Debug, Clone, Default)]
pub struct ChatCompletionsCompat {
    pub supports_reasoning_effort: bool,
    pub supports_developer_role: bool,
    pub requires_tool_result_name: bool,
    pub reasoning_field: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ChatCompletionsBackend {
    client: Client,
    base_url: String,
    compat: ChatCompletionsCompat,
    name: &'static str,
}

impl ChatCompletionsBackend {
    pub fn new(name: &'static str) -> Self {
        Self {
            client: Client::new(),
            base_url: DEFAULT_CHAT_COMPLETIONS_URL.to_string(),
            compat: ChatCompletionsCompat::default(),
            name,
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_compat(mut self, compat: ChatCompletionsCompat) -> Self {
        self.compat = compat;
        self
    }

    pub fn build_request_body(&self, request: &ModelRequest) -> serde_json::Value {
        let messages = request
            .messages
            .iter()
            .map(|message| chat_message_from_llm(message, &self.compat))
            .collect::<Vec<_>>();

        let tools = request
            .tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    }
                })
            })
            .collect::<Vec<_>>();

        let mut body = serde_json::json!({
            "model": request.model,
            "messages": messages,
            "stream": true,
        });

        if !request.system.is_empty() {
            let role = if self.compat.supports_developer_role {
                "developer"
            } else {
                "system"
            };
            if let Some(array) = body.get_mut("messages").and_then(serde_json::Value::as_array_mut) {
                array.insert(
                    0,
                    serde_json::json!({
                        "role": role,
                        "content": request.system,
                    }),
                );
            }
        }

        if !tools.is_empty() {
            body["tools"] = serde_json::Value::Array(tools);
            body["tool_choice"] = serde_json::json!("auto");
        }

        if let Some(temperature) = request.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }

        if self.compat.supports_reasoning_effort {
            body["reasoning_effort"] = serde_json::json!("medium");
        }

        body
    }
}

#[async_trait]
impl Backend for ChatCompletionsBackend {
    fn name(&self) -> &'static str {
        self.name
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
            .client
            .post(&self.base_url)
            .header("content-type", "application/json");
        if !request.api_key.expose_secret().trim().is_empty() {
            request_builder = request_builder.bearer_auth(request.api_key.expose_secret());
        }
        let response = request_builder
            .json(&body)
            .send()
            .await
            .with_context(|| format!("failed to send request to {}", self.base_url))?
            .error_for_status()
            .with_context(|| format!("chat completions request failed for {}", self.base_url))?;

        let compat = self.compat.clone();
        let stream = try_stream! {
            let mut buffer = String::new();
            let mut bytes = response.bytes_stream();
            let mut tool_calls: HashMap<u64, StreamingToolCall> = HashMap::new();
            let mut saw_tool_calls = false;
            let mut final_stop_reason = StopReason::EndTurn;

            while let Some(chunk) = bytes.next().await {
                let chunk = chunk.context("failed to read chat completions response chunk")?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(frame) = next_sse_frame(&mut buffer) {
                    for event in parse_sse_data(&frame, &compat)? {
                        match event {
                            ChatCompletionsEvent::TextDelta(delta) => {
                                yield ModelEvent::TextDelta(delta);
                            }
                            ChatCompletionsEvent::ToolCallStart { index, id, name } => {
                                saw_tool_calls = true;
                                tool_calls.insert(index, StreamingToolCall {
                                    id: id.clone(),
                                    name,
                                });
                                yield ModelEvent::ToolCallStart {
                                    id: id.clone(),
                                    name: tool_calls.get(&index).map(|call| call.name.clone()).unwrap_or_default(),
                                };
                            }
                            ChatCompletionsEvent::ToolCallArgsDelta { index, delta } => {
                                if let Some(call) = tool_calls.get(&index) {
                                    yield ModelEvent::ToolCallArgsDelta {
                                        id: call.id.clone(),
                                        delta,
                                    };
                                }
                            }
                            ChatCompletionsEvent::StopReason(stop_reason) => {
                                final_stop_reason = stop_reason;
                            }
                            ChatCompletionsEvent::Done => {
                                for (_, tool_call) in tool_calls.drain() {
                                    yield ModelEvent::ToolCallEnd { id: tool_call.id };
                                }

                                let stop_reason = if saw_tool_calls && matches!(final_stop_reason, StopReason::EndTurn) {
                                    StopReason::ToolCalls
                                } else {
                                    final_stop_reason.clone()
                                };
                                yield ModelEvent::Completed { stop_reason };
                            }
                            ChatCompletionsEvent::Error(message) => {
                                yield ModelEvent::Error(message);
                            }
                            ChatCompletionsEvent::Ignored => {}
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

#[derive(Debug, Clone)]
struct StreamingToolCall {
    id: String,
    name: String,
}

#[derive(Debug)]
enum ChatCompletionsEvent {
    TextDelta(String),
    ToolCallStart { index: u64, id: String, name: String },
    ToolCallArgsDelta { index: u64, delta: String },
    StopReason(StopReason),
    Done,
    Error(String),
    Ignored,
}

fn chat_message_from_llm(
    message: &crate::LlmMessage,
    compat: &ChatCompletionsCompat,
) -> serde_json::Value {
    match message.role {
        LlmRole::System => serde_json::json!({
            "role": if compat.supports_developer_role { "developer" } else { "system" },
            "content": flatten_text_parts(&message.parts),
        }),
        LlmRole::User => serde_json::json!({
            "role": "user",
            "content": flatten_text_parts(&message.parts),
        }),
        LlmRole::Assistant => {
            let tool_calls = message
                .parts
                .iter()
                .filter_map(|part| match part {
                    MessagePart::ToolCall(call) => Some(serde_json::json!({
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments_json,
                        }
                    })),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let text = flatten_text_parts(&message.parts);

            if tool_calls.is_empty() {
                serde_json::json!({
                    "role": "assistant",
                    "content": text,
                })
            } else {
                serde_json::json!({
                    "role": "assistant",
                    "content": if text.is_empty() { serde_json::Value::Null } else { serde_json::Value::String(text) },
                    "tool_calls": tool_calls,
                })
            }
        }
        LlmRole::Tool => {
            let tool_result = message.parts.iter().find_map(|part| match part {
                MessagePart::ToolResult(result) => Some(result),
                _ => None,
            });

            if let Some(result) = tool_result {
                let mut value = serde_json::json!({
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": result.content,
                });
                if compat.requires_tool_result_name {
                    value["name"] = serde_json::Value::String("tool".to_string());
                }
                value
            } else {
                serde_json::json!({
                    "role": "tool",
                    "content": flatten_text_parts(&message.parts),
                })
            }
        }
    }
}

fn flatten_text_parts(parts: &[MessagePart]) -> String {
    parts.iter()
        .filter_map(|part| match part {
            MessagePart::Text(text) => Some(text.text.as_str()),
            MessagePart::ToolCall(_) | MessagePart::ToolResult(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
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

fn parse_sse_data(
    frame: &str,
    compat: &ChatCompletionsCompat,
) -> anyhow::Result<Vec<ChatCompletionsEvent>> {
    let mut data_lines = Vec::new();

    for line in frame.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.trim_start());
        }
    }

    if data_lines.is_empty() {
        return Ok(Vec::new());
    }

    let data = data_lines.join("\n");
    if data == "[DONE]" {
        return Ok(vec![ChatCompletionsEvent::Done]);
    }

    let value: serde_json::Value = serde_json::from_str(&data)
        .with_context(|| format!("failed to parse SSE payload: {data}"))?;
    let choice = value
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .and_then(|choices| choices.first())
        .cloned()
        .unwrap_or_default();

    if let Some(message) = value
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(serde_json::Value::as_str)
    {
        return Ok(vec![ChatCompletionsEvent::Error(message.to_string())]);
    }

    let mut events = Vec::new();

    if let Some(reason) = choice.get("finish_reason").and_then(serde_json::Value::as_str) {
        events.push(ChatCompletionsEvent::StopReason(map_stop_reason(reason)));
    }

    let delta = choice.get("delta").cloned().unwrap_or_default();
    if let Some(text) = delta.get("content").and_then(serde_json::Value::as_str) {
        if !text.is_empty() {
            events.push(ChatCompletionsEvent::TextDelta(text.to_string()));
        }
    }

    if let Some(reasoning_field) = compat.reasoning_field.as_deref() {
        if let Some(reasoning) = delta.get(reasoning_field).and_then(serde_json::Value::as_str) {
            if !reasoning.is_empty() {
                events.push(ChatCompletionsEvent::TextDelta(reasoning.to_string()));
            }
        }
    }

    if let Some(tool_calls) = delta.get("tool_calls").and_then(serde_json::Value::as_array) {
        for tool_call in tool_calls {
            let index = tool_call
                .get("index")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_default();
            if let Some(id) = tool_call.get("id").and_then(serde_json::Value::as_str) {
                events.push(ChatCompletionsEvent::ToolCallStart {
                    index,
                    id: id.to_string(),
                    name: tool_call
                        .get("function")
                        .and_then(|function| function.get("name"))
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                });
            }
            if let Some(arguments) = tool_call
                .get("function")
                .and_then(|function| function.get("arguments"))
                .and_then(serde_json::Value::as_str)
            {
                events.push(ChatCompletionsEvent::ToolCallArgsDelta {
                    index,
                    delta: arguments.to_string(),
                });
            }
        }
    }

    if events.is_empty() {
        events.push(ChatCompletionsEvent::Ignored);
    }

    Ok(events)
}

fn map_stop_reason(reason: &str) -> StopReason {
    match reason {
        "tool_calls" => StopReason::ToolCalls,
        "length" => StopReason::MaxTokens,
        "content_filter" => StopReason::Error,
        _ => StopReason::EndTurn,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LlmMessage, MessagePart, ModelRequest, TextPart, ToolResultPart};
    use secrecy::SecretString;

    fn sample_request() -> ModelRequest {
        ModelRequest {
            system: "You are a test agent.".to_string(),
            messages: vec![
                LlmMessage {
                    role: LlmRole::User,
                    parts: vec![MessagePart::Text(TextPart {
                        text: "hello".to_string(),
                    })],
                },
                LlmMessage {
                    role: LlmRole::Tool,
                    parts: vec![MessagePart::ToolResult(ToolResultPart {
                        call_id: "call_1".to_string(),
                        content: "ok".to_string(),
                        is_error: false,
                    })],
                },
            ],
            tools: Vec::new(),
            model: "test-model".to_string(),
            api_key: SecretString::new("test-key".to_string().into_boxed_str()),
            temperature: Some(0.2),
        }
    }

    #[test]
    fn request_body_uses_developer_role_when_supported() {
        let backend = ChatCompletionsBackend::new("compatible").with_compat(ChatCompletionsCompat {
            supports_reasoning_effort: true,
            supports_developer_role: true,
            requires_tool_result_name: false,
            reasoning_field: Some("reasoning".to_string()),
        });

        let body = backend.build_request_body(&sample_request());
        let messages = body
            .get("messages")
            .and_then(serde_json::Value::as_array)
            .expect("messages array");
        assert_eq!(messages[0].get("role").and_then(serde_json::Value::as_str), Some("developer"));
        assert_eq!(body.get("reasoning_effort").and_then(serde_json::Value::as_str), Some("medium"));
    }

    #[test]
    fn request_body_adds_tool_name_when_required() {
        let backend = ChatCompletionsBackend::new("compat").with_compat(ChatCompletionsCompat {
            supports_reasoning_effort: false,
            supports_developer_role: false,
            requires_tool_result_name: true,
            reasoning_field: None,
        });

        let body = backend.build_request_body(&sample_request());
        let messages = body
            .get("messages")
            .and_then(serde_json::Value::as_array)
            .expect("messages array");
        let tool_message = messages.last().expect("tool message");
        assert_eq!(tool_message.get("role").and_then(serde_json::Value::as_str), Some("tool"));
        assert_eq!(tool_message.get("name").and_then(serde_json::Value::as_str), Some("tool"));
    }

    #[test]
    fn parser_handles_text_delta() {
        let frame = r#"data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}"#;
        let event = parse_sse_data(frame, &ChatCompletionsCompat::default())
            .expect("parse ok")
            .into_iter()
            .next()
            .expect("event");
        match event {
            ChatCompletionsEvent::TextDelta(delta) => assert_eq!(delta, "hello"),
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn parser_handles_reasoning_field() {
        let frame = r#"data: {"choices":[{"delta":{"reasoning":"thinking"},"finish_reason":null}]}"#;
        let event = parse_sse_data(
            frame,
            &ChatCompletionsCompat {
                supports_reasoning_effort: true,
                supports_developer_role: true,
                requires_tool_result_name: false,
                reasoning_field: Some("reasoning".to_string()),
            },
        )
        .expect("parse ok")
        .into_iter()
        .next()
        .expect("event");
        match event {
            ChatCompletionsEvent::TextDelta(delta) => assert_eq!(delta, "thinking"),
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn parser_handles_reasoning_content_field() {
        let frame = r#"data: {"choices":[{"delta":{"reasoning_content":"internal"},"finish_reason":null}]}"#;
        let event = parse_sse_data(
            frame,
            &ChatCompletionsCompat {
                supports_reasoning_effort: false,
                supports_developer_role: false,
                requires_tool_result_name: false,
                reasoning_field: Some("reasoning_content".to_string()),
            },
        )
        .expect("parse ok")
        .into_iter()
        .next()
        .expect("event");
        match event {
            ChatCompletionsEvent::TextDelta(delta) => assert_eq!(delta, "internal"),
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn parser_handles_tool_call_start() {
        let frame = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read"}}]},"finish_reason":null}]}"#;
        let event = parse_sse_data(frame, &ChatCompletionsCompat::default())
            .expect("parse ok")
            .into_iter()
            .next()
            .expect("event");
        match event {
            ChatCompletionsEvent::ToolCallStart { index, id, name } => {
                assert_eq!(index, 0);
                assert_eq!(id, "call_1");
                assert_eq!(name, "read");
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn parser_handles_tool_call_arguments_delta() {
        let frame = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":\"a\"}"}}]},"finish_reason":null}]}"#;
        let event = parse_sse_data(frame, &ChatCompletionsCompat::default())
            .expect("parse ok")
            .into_iter()
            .next()
            .expect("event");
        match event {
            ChatCompletionsEvent::ToolCallArgsDelta { index, delta } => {
                assert_eq!(index, 0);
                assert_eq!(delta, "{\"path\":\"a\"}");
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn parser_maps_tool_calls_finish_reason() {
        let frame = r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#;
        let event = parse_sse_data(frame, &ChatCompletionsCompat::default())
            .expect("parse ok")
            .into_iter()
            .next()
            .expect("event");
        match event {
            ChatCompletionsEvent::StopReason(reason) => assert_eq!(reason, StopReason::ToolCalls),
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn parser_handles_done_sentinel() {
        let event = parse_sse_data("data: [DONE]", &ChatCompletionsCompat::default())
            .expect("parse ok")
            .into_iter()
            .next()
            .expect("event");
        match event {
            ChatCompletionsEvent::Done => {}
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn parser_returns_multiple_events_from_single_frame() {
        let frame = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read","arguments":"{\"path\":\"a\"}"}},{"index":1,"id":"call_2","function":{"name":"write","arguments":"{\"path\":\"b\"}"}}]},"finish_reason":null}]}"#;
        let events = parse_sse_data(frame, &ChatCompletionsCompat::default()).expect("parse ok");
        assert_eq!(events.len(), 4);

        match &events[0] {
            ChatCompletionsEvent::ToolCallStart { index, id, name } => {
                assert_eq!((*index, id.as_str(), name.as_str()), (0, "call_1", "read"));
            }
            other => panic!("unexpected event: {other:?}"),
        }
        match &events[1] {
            ChatCompletionsEvent::ToolCallArgsDelta { index, delta } => {
                assert_eq!((*index, delta.as_str()), (0, "{\"path\":\"a\"}"));
            }
            other => panic!("unexpected event: {other:?}"),
        }
        match &events[2] {
            ChatCompletionsEvent::ToolCallStart { index, id, name } => {
                assert_eq!((*index, id.as_str(), name.as_str()), (1, "call_2", "write"));
            }
            other => panic!("unexpected event: {other:?}"),
        }
        match &events[3] {
            ChatCompletionsEvent::ToolCallArgsDelta { index, delta } => {
                assert_eq!((*index, delta.as_str()), (1, "{\"path\":\"b\"}"));
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn next_frame_handles_double_newline() {
        let mut buffer = "data: {\"choices\":[]}\n\nrest".to_string();
        let frame = next_sse_frame(&mut buffer).expect("frame");
        assert_eq!(frame, "data: {\"choices\":[]}");
        assert_eq!(buffer, "rest");
    }

    fn read_fixture(name: &str) -> &'static str {
        match name {
            "reasoning_field.sse" => include_str!("../tests/fixtures/reasoning_field.sse"),
            "reasoning_content_field.sse" => {
                include_str!("../tests/fixtures/reasoning_content_field.sse")
            }
            "multi_tool_calls.sse" => include_str!("../tests/fixtures/multi_tool_calls.sse"),
            "tool_calls_finish_reason.sse" => {
                include_str!("../tests/fixtures/tool_calls_finish_reason.sse")
            }
            "done.sse" => include_str!("../tests/fixtures/done.sse"),
            "error.sse" => include_str!("../tests/fixtures/error.sse"),
            other => panic!("unknown fixture: {other}"),
        }
    }

    #[test]
    fn reasoning_field_fixture_emits_reasoning_and_text() {
        let frame = read_fixture("reasoning_field.sse");
        let events = parse_sse_data(
            frame.trim_end(),
            &ChatCompletionsCompat {
                supports_reasoning_effort: true,
                supports_developer_role: true,
                requires_tool_result_name: false,
                reasoning_field: Some("reasoning".to_string()),
            },
        )
        .expect("parse ok");

        assert_eq!(events.len(), 2);
        match &events[0] {
            ChatCompletionsEvent::TextDelta(delta) => assert_eq!(delta, "I'll inspect the file."),
            other => panic!("unexpected first event: {other:?}"),
        }
        match &events[1] {
            ChatCompletionsEvent::TextDelta(delta) => {
                assert_eq!(delta, "I should inspect the file first.")
            }
            other => panic!("unexpected second event: {other:?}"),
        }
    }

    #[test]
    fn reasoning_content_field_fixture_emits_reasoning_content_and_text() {
        let frame = read_fixture("reasoning_content_field.sse");
        let events = parse_sse_data(
            frame.trim_end(),
            &ChatCompletionsCompat {
                supports_reasoning_effort: false,
                supports_developer_role: false,
                requires_tool_result_name: false,
                reasoning_field: Some("reasoning_content".to_string()),
            },
        )
        .expect("parse ok");

        assert_eq!(events.len(), 2);
        match &events[0] {
            ChatCompletionsEvent::TextDelta(delta) => assert_eq!(delta, "Working on it."),
            other => panic!("unexpected first event: {other:?}"),
        }
        match &events[1] {
            ChatCompletionsEvent::TextDelta(delta) => assert_eq!(delta, "Need to reason locally."),
            other => panic!("unexpected second event: {other:?}"),
        }
    }

    #[test]
    fn multi_tool_calls_fixture_emits_all_tool_call_events() {
        let frame = read_fixture("multi_tool_calls.sse");
        let events = parse_sse_data(frame.trim_end(), &ChatCompletionsCompat::default())
            .expect("parse ok");

        assert_eq!(events.len(), 4);
        match &events[0] {
            ChatCompletionsEvent::ToolCallStart { index, id, name } => {
                assert_eq!((*index, id.as_str(), name.as_str()), (0, "call_1", "read"));
            }
            other => panic!("unexpected event: {other:?}"),
        }
        match &events[1] {
            ChatCompletionsEvent::ToolCallArgsDelta { index, delta } => {
                assert_eq!((*index, delta.as_str()), (0, "{\"path\":\"src/main.rs\"}"));
            }
            other => panic!("unexpected event: {other:?}"),
        }
        match &events[2] {
            ChatCompletionsEvent::ToolCallStart { index, id, name } => {
                assert_eq!((*index, id.as_str(), name.as_str()), (1, "call_2", "write"));
            }
            other => panic!("unexpected event: {other:?}"),
        }
        match &events[3] {
            ChatCompletionsEvent::ToolCallArgsDelta { index, delta } => {
                assert_eq!((*index, delta.as_str()), (1, "{\"path\":\"README.md\"}"));
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn finish_reason_fixture_maps_to_tool_calls() {
        let frame = read_fixture("tool_calls_finish_reason.sse");
        let events = parse_sse_data(frame.trim_end(), &ChatCompletionsCompat::default())
            .expect("parse ok");
        assert_eq!(events.len(), 1);
        match &events[0] {
            ChatCompletionsEvent::StopReason(reason) => assert_eq!(*reason, StopReason::ToolCalls),
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn done_fixture_maps_to_done_event() {
        let frame = read_fixture("done.sse");
        let events = parse_sse_data(frame.trim_end(), &ChatCompletionsCompat::default())
            .expect("parse ok");
        assert_eq!(events.len(), 1);
        match &events[0] {
            ChatCompletionsEvent::Done => {}
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn error_fixture_maps_to_error_event() {
        let frame = read_fixture("error.sse");
        let events = parse_sse_data(frame.trim_end(), &ChatCompletionsCompat::default())
            .expect("parse ok");
        assert_eq!(events.len(), 1);
        match &events[0] {
            ChatCompletionsEvent::Error(message) => {
                assert_eq!(message, "upstream provider error")
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn frame_reader_handles_multiple_fixture_frames() {
        let mut buffer = format!(
            "{}\n\n{}",
            read_fixture("reasoning_field.sse").trim_end(),
            read_fixture("done.sse").trim_end()
        );
        let frame1 = next_sse_frame(&mut buffer).expect("frame 1");
        assert!(frame1.contains("reasoning"));
        assert!(buffer.contains("data: [DONE]"));
    }
}

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use secrecy::SecretString;
use serde::{Deserialize, Serialize};

pub type ModelEventStream = Pin<Box<dyn Stream<Item = anyhow::Result<ModelEvent>> + Send>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Capability {
    StreamingText,
    ToolCalling,
    Reasoning,
    ImageInput,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextPart {
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallPart {
    pub id: String,
    pub call_id: String,
    pub name: String,
    pub arguments_json: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResultPart {
    pub call_id: String,
    pub content: String,
    pub is_error: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessagePart {
    Text(TextPart),
    ToolCall(ToolCallPart),
    ToolResult(ToolResultPart),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmMessage {
    pub role: LlmRole,
    pub parts: Vec<MessagePart>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    EndTurn,
    ToolCalls,
    MaxTokens,
    Cancelled,
    Error,
}

#[derive(Debug, Clone)]
pub struct ModelRequest {
    pub system: String,
    pub messages: Vec<LlmMessage>,
    pub tools: Vec<ToolSpec>,
    pub model: String,
    pub api_key: SecretString,
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelEvent {
    TextDelta(String),
    ToolCallStart { id: String, name: String },
    ToolCallArgsDelta { id: String, delta: String },
    ToolCallEnd { id: String },
    Usage(Usage),
    Completed { stop_reason: StopReason },
    Error(String),
}

#[async_trait]
pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;
    fn supports(&self, capability: Capability) -> bool;
    async fn stream(&self, request: ModelRequest) -> anyhow::Result<ModelEventStream>;
}

use agent_model::{Backend, LlmMessage, LlmRole, MessagePart, ModelEvent, ModelRequest, TextPart};
use agent_session::{CompactionResult, SessionStore};
use futures::StreamExt;
use secrecy::SecretString;

pub(crate) async fn maybe_auto_compact<B: Backend>(
    session: &mut SessionStore,
    threshold: usize,
    keep_last: usize,
    api_key: Option<String>,
    backend: &B,
    summary_model: &str,
) -> anyhow::Result<Option<CompactionResult>> {
    if threshold == 0 || session.messages().len() < threshold {
        return Ok(None);
    }

    compact_session(session, keep_last, api_key, backend, summary_model).await
}

pub(crate) async fn compact_session<B: Backend>(
    session: &mut SessionStore,
    keep_last: usize,
    api_key: Option<String>,
    backend: &B,
    summary_model: &str,
) -> anyhow::Result<Option<CompactionResult>> {
    let Some(api_key) = api_key else {
        return session.compact_leaf(keep_last);
    };

    let compactable_messages = session.compactable_messages(keep_last);
    if compactable_messages.is_empty() {
        return Ok(None);
    }

    let summary = summarize_messages_for_compaction(
        backend,
        api_key,
        summary_model,
        &compactable_messages,
    )
    .await
    .unwrap_or_else(|_| summarize_messages_locally(&compactable_messages));
    session.compact_leaf_with_summary(keep_last, summary)
}

async fn summarize_messages_for_compaction<B: Backend>(
    backend: &B,
    api_key: String,
    model: &str,
    messages: &[agent_core::AgentMessage],
) -> anyhow::Result<String> {
    let transcript = messages
        .iter()
        .map(render_summary_message_line)
        .collect::<Vec<_>>()
        .join("\n");
    let request = ModelRequest {
        system: "Summarize the earlier coding-agent conversation for future continuation. Preserve user intent, decisions, constraints, open tasks, file paths, and unfinished tool results. Be concise and structured for another agent.".to_string(),
        messages: vec![LlmMessage {
            role: LlmRole::User,
            parts: vec![MessagePart::Text(TextPart {
                text: format!("Summarize this transcript for compaction:\n\n{transcript}"),
            })],
        }],
        tools: Vec::new(),
        model: model.to_string(),
        api_key: SecretString::new(api_key.into_boxed_str()),
        temperature: None,
    };

    let mut stream = backend.stream(request).await?;
    let mut summary = String::new();
    while let Some(event) = stream.next().await {
        match event? {
            ModelEvent::TextDelta(delta) => summary.push_str(&delta),
            ModelEvent::Completed { .. } => break,
            ModelEvent::Error(message) => anyhow::bail!("summary request failed: {message}"),
            ModelEvent::ToolCallStart { .. }
            | ModelEvent::ToolCallArgsDelta { .. }
            | ModelEvent::ToolCallEnd { .. }
            | ModelEvent::Usage(_) => {}
        }
    }

    let summary = summary.trim().to_string();
    if summary.is_empty() {
        anyhow::bail!("summary request returned empty text");
    }
    Ok(summary)
}

fn summarize_messages_locally(messages: &[agent_core::AgentMessage]) -> String {
    messages
        .iter()
        .map(render_summary_message_line)
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_summary_message_line(message: &agent_core::AgentMessage) -> String {
    let role = match message {
        agent_core::AgentMessage::User(_) => "user",
        agent_core::AgentMessage::Assistant(_) => "assistant",
        agent_core::AgentMessage::ToolResult(_) => "tool",
    };
    let mut line = String::new();
    for part in &message.as_llm_message().parts {
        match part {
            MessagePart::Text(text) => {
                if !line.is_empty() {
                    line.push(' ');
                }
                line.push_str(text.text.trim());
            }
            MessagePart::ToolCall(call) => {
                if !line.is_empty() {
                    line.push(' ');
                }
                line.push_str(&format!("tool_call {} {}", call.name, call.arguments_json));
            }
            MessagePart::ToolResult(result) => {
                if !line.is_empty() {
                    line.push(' ');
                }
                line.push_str(&format!(
                    "tool_result {} {}",
                    result.call_id,
                    result.content.trim()
                ));
            }
        }
    }
    format!("- {role}: {line}")
}

use agent_core::AgentMessage;

use crate::{BlockKind, LineKind, RenderBlock, StyledLine};

pub(crate) fn format_message(message: &AgentMessage) -> Vec<StyledLine> {
    let llm = message.as_llm_message();
    let mut lines = Vec::new();

    for part in &llm.parts {
        match part {
            agent_model::MessagePart::Text(text) => match message {
                AgentMessage::User(_) => {
                    lines.extend(message_lines(crate::USER_PREFIX, &text.text, LineKind::User))
                }
                AgentMessage::Assistant(_) => {
                    lines.extend(message_lines("", &text.text, LineKind::Plain))
                }
                AgentMessage::ToolResult(_) => {
                    lines.extend(message_lines("  ", &text.text, LineKind::Tool))
                }
            },
            agent_model::MessagePart::ToolCall(call) => {
                lines.push(StyledLine::new(
                    format!("· tool call {}", call.name),
                    LineKind::Tool,
                ));
                lines.extend(message_lines("  ", &call.arguments_json, LineKind::Tool));
            }
            agent_model::MessagePart::ToolResult(result) => {
                lines.push(StyledLine::new(
                    format!("· tool result {}", result.call_id),
                    LineKind::Tool,
                ));
                lines.extend(message_lines("  ", &result.content, LineKind::Tool));
            }
        }
    }

    if lines.is_empty() {
        lines.push(StyledLine::blank());
    }

    lines
}

pub(crate) fn message_lines(prefix: &str, text: &str, kind: LineKind) -> Vec<StyledLine> {
    let mut lines = Vec::new();
    for line in text.split('\n') {
        if prefix.is_empty() {
            lines.push(StyledLine::new(line, kind));
        } else {
            lines.push(StyledLine::new(format!("{prefix}{line}"), kind));
        }
    }
    if lines.is_empty() {
        lines.push(StyledLine::new(prefix, kind));
    }
    lines
}

pub(crate) fn append_text_to_block(block: &mut RenderBlock, prefix: &str, delta: &str) {
    if block.lines.is_empty() {
        let kind = if prefix.trim_start().starts_with('·') || prefix.starts_with("  ") {
            LineKind::Tool
        } else {
            LineKind::Plain
        };
        block.lines.push(StyledLine::new(prefix, kind));
    }

    let mut first = true;
    for segment in delta.split_inclusive('\n') {
        let ends_with_newline = segment.ends_with('\n');
        let clean = segment.trim_end_matches('\n');

        if first {
            if let Some(last) = block.lines.last_mut() {
                last.text.push_str(clean);
            }
            first = false;
        } else {
            let kind = block.lines.last().map(|line| line.kind).unwrap_or(LineKind::Plain);
            block.lines.push(StyledLine::new(format!("{prefix}{clean}"), kind));
        }

        if ends_with_newline {
            let kind = block.lines.last().map(|line| line.kind).unwrap_or(LineKind::Plain);
            block.lines.push(StyledLine::new(prefix, kind));
        }
    }
}

pub(crate) fn classify_block_from_message(message: &AgentMessage) -> BlockKind {
    match message {
        AgentMessage::ToolResult(_) => BlockKind::Tool,
        AgentMessage::User(_) | AgentMessage::Assistant(_) => BlockKind::Conversation,
    }
}

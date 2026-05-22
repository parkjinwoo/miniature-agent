use agent_core::AgentMessage;

use crate::{BlockKind, LineKind, RenderBlock, StyledLine};

pub(crate) fn format_message(message: &AgentMessage) -> Vec<StyledLine> {
    let llm = message.as_llm_message();
    let mut lines = Vec::new();

    for part in &llm.parts {
        match part {
            agent_model::MessagePart::Text(text) => match message {
                AgentMessage::User(_) => lines.extend(message_lines(
                    crate::USER_PREFIX,
                    &text.text,
                    LineKind::User,
                )),
                AgentMessage::Assistant(_) => {
                    lines.extend(message_lines("", &text.text, LineKind::Assistant))
                }
                AgentMessage::ToolResult(_) => lines.extend(technical_lines("  ", &text.text)),
            },
            agent_model::MessagePart::ToolCall(call) => {
                lines.push(StyledLine::new(
                    format!("· tool call {}", call.name),
                    LineKind::ToolTitle,
                ));
                lines.extend(technical_lines("  ", &call.arguments_json));
            }
            agent_model::MessagePart::ToolResult(result) => {
                lines.push(StyledLine::new(
                    format!("· tool result {}", result.call_id),
                    LineKind::ToolTitle,
                ));
                lines.extend(technical_lines("  ", &result.content));
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

fn technical_lines(prefix: &str, text: &str) -> Vec<StyledLine> {
    let mut lines = message_lines(prefix, text, LineKind::Tool);
    if lines.len() <= crate::MAX_TRANSCRIPT_TOOL_OUTPUT_LINES {
        return lines;
    }

    let omitted = lines.len() - crate::MAX_TRANSCRIPT_TOOL_OUTPUT_LINES;
    let mut tail = lines.split_off(omitted);
    tail.insert(
        0,
        StyledLine::new(
            format!("{prefix}... {omitted} earlier lines omitted"),
            LineKind::Tool,
        ),
    );
    tail
}

pub(crate) fn append_text_to_block(block: &mut RenderBlock, prefix: &str, delta: &str) {
    if block.lines.is_empty() {
        let kind = stream_line_kind(prefix);
        block.lines.push(StyledLine::new(prefix, kind));
    } else if !prefix.is_empty()
        && block
            .lines
            .last()
            .is_some_and(|line| !line.text.starts_with(prefix))
    {
        block
            .lines
            .push(StyledLine::new(prefix, stream_line_kind(prefix)));
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
            let kind = block
                .lines
                .last()
                .map(|line| line.kind)
                .unwrap_or(LineKind::Assistant);
            block
                .lines
                .push(StyledLine::new(format!("{prefix}{clean}"), kind));
        }

        if ends_with_newline {
            let kind = block
                .lines
                .last()
                .map(|line| line.kind)
                .unwrap_or(LineKind::Assistant);
            block.lines.push(StyledLine::new(prefix, kind));
        }
    }
}

fn stream_line_kind(prefix: &str) -> LineKind {
    if prefix.trim_start().starts_with('·') {
        LineKind::ToolTitle
    } else if prefix.starts_with("  ") {
        LineKind::Tool
    } else {
        LineKind::Assistant
    }
}

pub(crate) fn classify_block_from_message(message: &AgentMessage) -> BlockKind {
    match message {
        AgentMessage::ToolResult(_) => BlockKind::Tool,
        AgentMessage::User(_) | AgentMessage::Assistant(_) => BlockKind::Conversation,
    }
}

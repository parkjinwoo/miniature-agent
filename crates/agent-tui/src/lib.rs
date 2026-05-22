use std::io::{stdout, Stdout};
use std::time::{Duration, Instant};

use agent_core::{AgentEvent, AgentMessage};
use agent_model::Usage;
use anyhow::Context;
use crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers, KeyboardEnhancementFlags, MouseEvent, MouseEventKind,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::Position;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Paragraph, Widget};
use ratatui::{Frame, Terminal, TerminalOptions, Viewport};
use time::OffsetDateTime;
use time::UtcOffset;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use bottom_pane::{cursor_position_for_input, render_input_lines, wrap_line_by_display_width};
use conversation::{
    append_text_to_block, classify_block_from_message, format_message, message_lines,
};

mod bottom_pane;
mod conversation;

const MIN_INLINE_VIEWPORT_HEIGHT: u16 = 10;
const DEFAULT_INLINE_VIEWPORT_HEIGHT: u16 = 14;
const MAX_INLINE_VIEWPORT_HEIGHT: u16 = 18;
const INPUT_PREFIX: &str = "> ";
const INPUT_CONTINUATION_PREFIX: &str = "  ";
const INPUT_PREFIX_WIDTH: usize = 2;
const SELECTION_PREVIEW_MIN_WIDTH: usize = 72;
const MAX_TRANSCRIPT_TOOL_OUTPUT_LINES: usize = 8;
const USER_PREFIX: &str = "▌  ";

type RatTerminal = Terminal<CrosstermBackend<Stdout>>;

pub struct TuiApp {
    terminal: RatTerminal,
    state: TuiState,
    printed_committed_blocks: usize,
    entered: bool,
}

impl TuiApp {
    pub fn new() -> anyhow::Result<Self> {
        let backend = CrosstermBackend::new(stdout());
        let (viewport_width, viewport_height) = initial_viewport_size();
        let terminal = Terminal::with_options(
            backend,
            TerminalOptions {
                viewport: Viewport::Inline(viewport_height),
            },
        )?;

        Ok(Self {
            terminal,
            state: TuiState {
                viewport_width,
                viewport_height,
                ..Default::default()
            },
            printed_committed_blocks: 0,
            entered: false,
        })
    }

    pub fn enter(&mut self) -> anyhow::Result<()> {
        if self.entered {
            return Ok(());
        }

        enable_raw_mode().context("failed to enable raw mode")?;
        execute!(
            std::io::stdout(),
            EnableBracketedPaste,
            PushKeyboardEnhancementFlags(
                KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                    | KeyboardEnhancementFlags::REPORT_EVENT_TYPES
                    | KeyboardEnhancementFlags::REPORT_ALTERNATE_KEYS
            )
        )
        .ok();
        self.entered = true;
        self.render()
    }

    pub fn leave(&mut self) -> anyhow::Result<()> {
        if !self.entered {
            return Ok(());
        }

        self.terminal.show_cursor().ok();
        execute!(
            std::io::stdout(),
            DisableBracketedPaste,
            PopKeyboardEnhancementFlags
        )
        .ok();
        disable_raw_mode().context("failed to disable raw mode")?;
        self.entered = false;
        Ok(())
    }

    pub fn push_event(&mut self, event: AgentEvent) {
        self.state.apply_event(event);
    }

    pub fn push_message(&mut self, message: &AgentMessage) {
        self.state.push_message(message);
    }

    pub fn push_system_note(&mut self, note: impl Into<String>) {
        self.state.push_system_note(note);
    }

    pub fn push_user_input(&mut self, text: &str) {
        self.state.push_user_input(text);
    }

    pub fn replace_messages(&mut self, messages: &[AgentMessage]) {
        self.state.replace_messages(messages);
        self.printed_committed_blocks = 0;
    }

    pub fn set_status(&mut self, status: impl Into<String>) {
        self.state.set_status(status);
    }

    pub fn set_footer_context(
        &mut self,
        path_label: impl Into<String>,
        model_label: impl Into<String>,
    ) {
        self.state.footer_path = path_label.into();
        self.state.footer_model = model_label.into();
    }

    pub fn redraw(&mut self) -> anyhow::Result<()> {
        self.render()
    }

    pub fn prompt_once(&mut self) -> anyhow::Result<Option<String>> {
        loop {
            match self.poll_prompt_action(Duration::from_millis(50))? {
                Some(PromptAction::Submit(text)) => return Ok(Some(text)),
                Some(PromptAction::Quit) => return Ok(None),
                Some(PromptAction::Continue) | None => {}
            }
        }
    }

    pub fn pick_from_list(
        &mut self,
        title: impl Into<String>,
        items: &[String],
    ) -> anyhow::Result<Option<usize>> {
        self.pick_from_list_at(title, items, 0)
    }

    pub fn pick_from_list_at(
        &mut self,
        title: impl Into<String>,
        items: &[String],
        initial_index: usize,
    ) -> anyhow::Result<Option<usize>> {
        if items.is_empty() {
            return Ok(None);
        }

        self.state.selection_title = Some(title.into());
        self.state.selection_items = items.to_vec();
        self.state.selection_index = initial_index.min(items.len().saturating_sub(1));

        loop {
            self.render()?;
            if !event::poll(Duration::from_millis(50)).context("failed to poll terminal events")? {
                continue;
            }

            match event::read().context("failed to read terminal event")? {
                Event::Key(key) if should_handle_key_event(key) => match key.code {
                    KeyCode::Esc => {
                        self.clear_selection();
                        return Ok(None);
                    }
                    KeyCode::Enter => {
                        let index = self
                            .state
                            .selection_index
                            .min(self.state.selection_items.len().saturating_sub(1));
                        self.clear_selection();
                        return Ok(Some(index));
                    }
                    KeyCode::Up => {
                        self.state.selection_index = self.state.selection_index.saturating_sub(1);
                    }
                    KeyCode::Down => {
                        let max = self.state.selection_items.len().saturating_sub(1);
                        self.state.selection_index = (self.state.selection_index + 1).min(max);
                    }
                    KeyCode::PageUp => {
                        let page_size = selection_page_size(self.state.viewport_height);
                        self.state.selection_index =
                            self.state.selection_index.saturating_sub(page_size);
                    }
                    KeyCode::PageDown => {
                        let max = self.state.selection_items.len().saturating_sub(1);
                        let page_size = selection_page_size(self.state.viewport_height);
                        self.state.selection_index =
                            (self.state.selection_index + page_size).min(max);
                    }
                    _ => {}
                },
                Event::Mouse(mouse) => self.handle_mouse_in_selection(mouse),
                Event::Resize(width, height) => {
                    self.state.viewport_width = width;
                    self.state.viewport_height = height;
                }
                _ => {}
            }
        }
    }

    pub fn poll_running_action(&mut self, timeout: Duration) -> anyhow::Result<RunningAction> {
        self.render()?;
        if !event::poll(timeout).context("failed to poll terminal events")? {
            return Ok(RunningAction::Continue);
        }

        match event::read().context("failed to read terminal event")? {
            Event::Key(key) if should_handle_key_event(key) => Ok(self.handle_running_key(key)),
            Event::Paste(text) => {
                self.state.insert_text(&text);
                Ok(RunningAction::Continue)
            }
            Event::Mouse(mouse) => {
                self.handle_mouse(mouse);
                Ok(RunningAction::Continue)
            }
            Event::Resize(width, height) => {
                self.state.viewport_width = width;
                self.state.viewport_height = height;
                Ok(RunningAction::Continue)
            }
            _ => Ok(RunningAction::Continue),
        }
    }

    pub fn poll_prompt_action(
        &mut self,
        timeout: Duration,
    ) -> anyhow::Result<Option<PromptAction>> {
        self.render()?;
        if !event::poll(timeout).context("failed to poll terminal events")? {
            return Ok(None);
        }

        match event::read().context("failed to read terminal event")? {
            Event::Key(key) if should_handle_key_event(key) => Ok(self.handle_key(key)),
            Event::Paste(text) => {
                self.state.insert_text(&text);
                Ok(Some(PromptAction::Continue))
            }
            Event::Mouse(mouse) => {
                self.handle_mouse(mouse);
                Ok(Some(PromptAction::Continue))
            }
            Event::Resize(width, height) => {
                self.state.viewport_width = width;
                self.state.viewport_height = height;
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    fn clear_selection(&mut self) {
        self.state.selection_title = None;
        self.state.selection_items.clear();
        self.state.selection_index = 0;
    }

    fn handle_key(&mut self, key: KeyEvent) -> Option<PromptAction> {
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                Some(PromptAction::Quit)
            }
            KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.state.insert_char('\n');
                Some(PromptAction::Continue)
            }
            KeyCode::Enter
                if key.modifiers.contains(KeyModifiers::SHIFT)
                    || key.modifiers.contains(KeyModifiers::ALT) =>
            {
                self.state.insert_char('\n');
                Some(PromptAction::Continue)
            }
            KeyCode::Enter => {
                let text = std::mem::take(&mut self.state.input);
                self.state.cursor = 0;
                if text.trim().is_empty() {
                    None
                } else {
                    Some(PromptAction::Submit(text))
                }
            }
            KeyCode::Backspace => {
                self.state.backspace();
                Some(PromptAction::Continue)
            }
            KeyCode::Delete => {
                self.state.delete();
                Some(PromptAction::Continue)
            }
            KeyCode::Left => {
                self.state.move_left();
                Some(PromptAction::Continue)
            }
            KeyCode::Right => {
                self.state.move_right();
                Some(PromptAction::Continue)
            }
            KeyCode::Home => {
                self.state.move_to_line_start();
                Some(PromptAction::Continue)
            }
            KeyCode::End => {
                self.state.move_to_line_end();
                Some(PromptAction::Continue)
            }
            KeyCode::Up => {
                self.state.move_vertical(-1);
                Some(PromptAction::Continue)
            }
            KeyCode::Down => {
                self.state.move_vertical(1);
                Some(PromptAction::Continue)
            }
            KeyCode::Char(ch) => {
                self.state.insert_char(ch);
                Some(PromptAction::Continue)
            }
            _ => Some(PromptAction::Continue),
        }
    }

    fn handle_running_key(&mut self, key: KeyEvent) -> RunningAction {
        handle_running_key_state(&mut self.state, key)
    }

    fn handle_mouse(&mut self, _mouse: MouseEvent) {}

    fn handle_mouse_in_selection(&mut self, mouse: MouseEvent) {
        match mouse.kind {
            MouseEventKind::ScrollUp => {
                self.state.selection_index = self.state.selection_index.saturating_sub(1);
            }
            MouseEventKind::ScrollDown => {
                let max = self.state.selection_items.len().saturating_sub(1);
                self.state.selection_index = (self.state.selection_index + 1).min(max);
            }
            _ => {}
        }
    }

    fn render(&mut self) -> anyhow::Result<()> {
        if !self.entered {
            return Ok(());
        }

        self.flush_committed_blocks()?;
        let state = &self.state;
        self.terminal.draw(|frame| render_frame(frame, state))?;
        Ok(())
    }

    fn flush_committed_blocks(&mut self) -> anyhow::Result<()> {
        if self.printed_committed_blocks >= self.state.committed_blocks.len() {
            return Ok(());
        }

        let size = self.terminal.size()?;
        let width = size.width as usize;
        let lines = self
            .state
            .render_committed_lines_from(width, self.printed_committed_blocks);

        for chunk in lines.chunks(u16::MAX as usize) {
            self.terminal.insert_before(chunk.len() as u16, |buf| {
                Paragraph::new(Text::from(ratatui_lines(chunk, buf.area.width as usize)))
                    .render(buf.area, buf);
            })?;
        }

        self.printed_committed_blocks = self.state.committed_blocks.len();
        Ok(())
    }
}

impl Drop for TuiApp {
    fn drop(&mut self) {
        let _ = self.leave();
    }
}

pub enum PromptAction {
    Submit(String),
    Quit,
    Continue,
}

pub enum RunningAction {
    Abort,
    Quit,
    Continue,
    QueueSubmit(String),
}

#[derive(Default)]
struct TuiState {
    committed_blocks: Vec<RenderBlock>,
    live_assistant: Option<RenderBlock>,
    live_tool: Option<RenderBlock>,
    input: String,
    cursor: usize,
    queued_input: Option<String>,
    status: String,
    status_since: Option<Instant>,
    footer_path: String,
    footer_model: String,
    latest_usage: Option<Usage>,
    viewport_width: u16,
    viewport_height: u16,
    selection_title: Option<String>,
    selection_items: Vec<String>,
    selection_index: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LineKind {
    Plain,
    Assistant,
    User,
    System,
    Tool,
    ToolTitle,
    Selection,
    Status,
    Hint,
    Input,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StyledLine {
    text: String,
    kind: LineKind,
}

impl StyledLine {
    fn new(text: impl Into<String>, kind: LineKind) -> Self {
        Self {
            text: text.into(),
            kind,
        }
    }

    fn blank() -> Self {
        Self::new("", LineKind::Plain)
    }
}

#[derive(Clone)]
struct RenderBlock {
    lines: Vec<StyledLine>,
    kind: BlockKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BottomSurface {
    lines: Vec<StyledLine>,
    cursor_row: usize,
    cursor_col: usize,
    live_rows: usize,
    input_rows: usize,
    info_rows: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SelectionSurface {
    lines: Vec<StyledLine>,
    selected_row: usize,
    has_preview: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockKind {
    Conversation,
    System,
    Tool,
}

impl TuiState {
    fn set_status(&mut self, status: impl Into<String>) {
        let status = status.into();
        if self.status != status {
            self.status = status;
            self.status_since = Some(Instant::now());
        } else if self.status_since.is_none() {
            self.status_since = Some(Instant::now());
        }
    }

    fn insert_char(&mut self, ch: char) {
        let byte = char_to_byte_index(&self.input, self.cursor);
        self.input.insert(byte, ch);
        self.cursor += 1;
    }

    fn insert_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }

        let byte = char_to_byte_index(&self.input, self.cursor);
        self.input.insert_str(byte, text);
        self.cursor += text.chars().count();
    }

    fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let end = char_to_byte_index(&self.input, self.cursor);
        let start = char_to_byte_index(&self.input, self.cursor - 1);
        self.input.replace_range(start..end, "");
        self.cursor -= 1;
    }

    fn delete(&mut self) {
        let total = self.input.chars().count();
        if self.cursor >= total {
            return;
        }
        let start = char_to_byte_index(&self.input, self.cursor);
        let end = char_to_byte_index(&self.input, self.cursor + 1);
        self.input.replace_range(start..end, "");
    }

    fn move_left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    fn move_right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.input.chars().count());
    }

    fn move_to_line_start(&mut self) {
        let chars: Vec<char> = self.input.chars().collect();
        let mut cursor = self.cursor.min(chars.len());
        while cursor > 0 && chars[cursor - 1] != '\n' {
            cursor -= 1;
        }
        self.cursor = cursor;
    }

    fn move_to_line_end(&mut self) {
        let chars: Vec<char> = self.input.chars().collect();
        let mut cursor = self.cursor.min(chars.len());
        while cursor < chars.len() && chars[cursor] != '\n' {
            cursor += 1;
        }
        self.cursor = cursor;
    }

    fn move_vertical(&mut self, direction: isize) {
        let chars: Vec<char> = self.input.chars().collect();
        let cursor = self.cursor.min(chars.len());
        let (row, col) = row_col_for_cursor(&chars, cursor);
        let target_row = if direction < 0 {
            row.saturating_sub(direction.unsigned_abs())
        } else {
            row.saturating_add(direction as usize)
        };
        self.cursor = cursor_for_row_col(&chars, target_row, col);
    }

    fn render_committed_lines_from(&self, width: usize, start_block: usize) -> Vec<StyledLine> {
        let mut raw_lines = Vec::new();
        for (index, block) in self.committed_blocks.iter().enumerate().skip(start_block) {
            if index > 0 && block.kind != BlockKind::System {
                raw_lines.push(StyledLine::blank());
            }
            raw_lines.extend(block.lines.iter().cloned());
        }
        wrap_lines(&raw_lines, width)
    }

    fn render_live_lines(&self, width: usize, max_lines: usize) -> Vec<StyledLine> {
        if self.live_assistant.is_none()
            && self.live_tool.is_none()
            && !self.status.starts_with("Running")
        {
            return Vec::new();
        }

        let mut lines = Vec::new();
        let title = self.live_title();
        if !title.is_empty() {
            lines.extend(
                wrap_line_by_display_width(&title, width)
                    .into_iter()
                    .map(|line| StyledLine::new(line, LineKind::Status)),
            );
        }

        if let Some(block) = &self.live_tool {
            let mut tool_lines = wrap_lines(&block.lines, width);
            if !tool_lines.is_empty() {
                lines.push(StyledLine::blank());
            }
            let tool_budget = live_tool_tail_budget(max_lines);
            let keep = tool_lines.len().saturating_sub(tool_budget);
            lines.extend(tool_lines.drain(keep..));
        }

        if let Some(block) = &self.live_assistant {
            let mut assistant_lines = wrap_lines(&block.lines, width);
            assistant_lines.retain(|line| !line.text.trim().is_empty());
            if !assistant_lines.is_empty() {
                if !lines.is_empty() {
                    lines.push(StyledLine::blank());
                }
                lines.extend(assistant_lines);
            }
        }

        if lines.len() > max_lines {
            lines.split_off(lines.len() - max_lines)
        } else {
            lines
        }
    }

    fn status_line(&self) -> String {
        let mut parts = Vec::new();
        if !self.status.is_empty() && self.status != "Ready" {
            if self.status.starts_with("Running") {
                parts.push("Working".to_string());
                if let Some(status_since) = self.status_since {
                    parts.push(format_elapsed(status_since.elapsed()));
                }
                parts.push("Esc abort".to_string());
            } else {
                parts.push(self.status.clone());
            }
        }
        parts.join(" · ")
    }

    fn info_line(&self) -> String {
        let status = self.status_line();
        let footer = self.footer_info_line();
        match (status.is_empty(), footer.is_empty()) {
            (false, false) => format!("{status} · {footer}"),
            (false, true) => status,
            (true, false) => footer,
            (true, true) => String::new(),
        }
    }

    fn footer_info_line(&self) -> String {
        let mut parts = Vec::new();
        if !self.footer_path.is_empty() {
            parts.push(self.footer_path.clone());
        }
        if !self.footer_model.is_empty() {
            parts.push(self.footer_model.clone());
        }
        if let Some(usage) = &self.latest_usage {
            let total = usage.input_tokens + usage.output_tokens;
            parts.push(format!("{} tok", format_token_count(total)));
        } else {
            parts.push("usage n/a".to_string());
        }
        parts.push(current_time_label());
        parts.join(" · ")
    }

    fn live_title(&self) -> String {
        let mut parts = Vec::new();
        if self.live_tool.is_some() {
            parts.push("Using tools".to_string());
        } else if self.live_assistant.is_some() || self.status.starts_with("Running") {
            parts.push("Composing reply".to_string());
        }

        if let Some(status_since) = self.status_since {
            parts.push(format_elapsed(status_since.elapsed()));
        }
        parts.join(" · ")
    }

    fn push_user_input(&mut self, text: &str) {
        self.committed_blocks.push(RenderBlock {
            lines: message_lines(USER_PREFIX, text, LineKind::User),
            kind: BlockKind::Conversation,
        });
        self.live_assistant = None;
        self.live_tool = None;
        self.queued_input = None;
    }

    fn push_message(&mut self, message: &AgentMessage) {
        self.committed_blocks.push(RenderBlock {
            lines: format_message(message),
            kind: classify_block_from_message(message),
        });
        self.live_assistant = None;
        self.live_tool = None;
    }

    fn push_system_note(&mut self, note: impl Into<String>) {
        let note = note.into();
        self.committed_blocks.push(RenderBlock {
            lines: message_lines("· ", &note, LineKind::System),
            kind: BlockKind::System,
        });
        self.live_assistant = None;
        self.live_tool = None;
    }

    fn replace_messages(&mut self, messages: &[AgentMessage]) {
        self.committed_blocks.clear();
        self.live_assistant = None;
        self.live_tool = None;
        self.queued_input = None;
        self.status_since = None;
        for message in messages {
            self.push_message(message);
        }
    }

    fn apply_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::AgentStart | AgentEvent::TurnStart => {}
            AgentEvent::MessageStart { role } => match role {
                agent_model::LlmRole::Assistant => {
                    self.live_assistant = Some(RenderBlock {
                        lines: vec![StyledLine::new("", LineKind::Assistant)],
                        kind: BlockKind::Conversation,
                    });
                }
                agent_model::LlmRole::Tool => {
                    self.live_tool = Some(RenderBlock {
                        lines: vec![StyledLine::new("· tool", LineKind::ToolTitle)],
                        kind: BlockKind::Tool,
                    });
                }
                agent_model::LlmRole::User | agent_model::LlmRole::System => {}
            },
            AgentEvent::TextDelta(delta) => {
                if let Some(block) = self.live_assistant.as_mut() {
                    append_text_to_block(block, "", &delta);
                } else {
                    self.live_assistant = Some(RenderBlock {
                        lines: vec![StyledLine::new(delta, LineKind::Assistant)],
                        kind: BlockKind::Conversation,
                    });
                }
            }
            AgentEvent::ToolCallStart { id, name } => {
                self.live_tool = Some(RenderBlock {
                    lines: vec![StyledLine::new(
                        format!("· tool {name} ({id})"),
                        LineKind::ToolTitle,
                    )],
                    kind: BlockKind::Tool,
                });
            }
            AgentEvent::ToolCallArgsDelta { delta, .. } => {
                if let Some(block) = self.live_tool.as_mut() {
                    append_text_to_block(block, "  ", &delta);
                }
            }
            AgentEvent::ToolCallEnd { id } => {
                if let Some(block) = self.live_tool.as_mut() {
                    block
                        .lines
                        .push(StyledLine::new(format!("  done {id}"), LineKind::ToolTitle));
                }
            }
            AgentEvent::Usage(usage) => {
                self.latest_usage = Some(usage);
            }
            AgentEvent::MessageEnd { message, .. } => match message.role {
                agent_model::LlmRole::Assistant => {
                    self.live_assistant = None;
                    self.committed_blocks.push(RenderBlock {
                        lines: format_message(&AgentMessage::Assistant(message)),
                        kind: BlockKind::Conversation,
                    });
                }
                agent_model::LlmRole::Tool => {
                    self.live_tool = None;
                    self.committed_blocks.push(RenderBlock {
                        lines: format_message(&AgentMessage::ToolResult(message)),
                        kind: BlockKind::Tool,
                    });
                }
                agent_model::LlmRole::User | agent_model::LlmRole::System => {}
            },
            AgentEvent::ToolResultReady { message } => {
                self.committed_blocks.push(RenderBlock {
                    lines: format_message(&AgentMessage::ToolResult(message)),
                    kind: BlockKind::Tool,
                });
                self.live_tool = None;
            }
            AgentEvent::TurnEnd { .. } => {}
            AgentEvent::AgentEnd => {
                self.live_assistant = None;
                self.live_tool = None;
            }
        }
    }
}

fn render_frame(frame: &mut Frame<'_>, state: &TuiState) {
    if state.selection_items.is_empty() {
        render_bottom_frame(frame, state);
    } else {
        render_selection_frame(frame, state);
    }
}

fn handle_running_key_state(state: &mut TuiState, key: KeyEvent) -> RunningAction {
    match key.code {
        KeyCode::Esc => RunningAction::Abort,
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => RunningAction::Quit,
        KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            state.insert_char('\n');
            RunningAction::Continue
        }
        KeyCode::Enter
            if key.modifiers.contains(KeyModifiers::SHIFT)
                || key.modifiers.contains(KeyModifiers::ALT) =>
        {
            state.insert_char('\n');
            RunningAction::Continue
        }
        KeyCode::Enter => {
            let text = std::mem::take(&mut state.input);
            state.cursor = 0;
            if text.trim().is_empty() {
                RunningAction::Continue
            } else {
                state.queued_input = Some(text.clone());
                RunningAction::QueueSubmit(text)
            }
        }
        KeyCode::Backspace => {
            state.backspace();
            RunningAction::Continue
        }
        KeyCode::Delete => {
            state.delete();
            RunningAction::Continue
        }
        KeyCode::Left => {
            state.move_left();
            RunningAction::Continue
        }
        KeyCode::Right => {
            state.move_right();
            RunningAction::Continue
        }
        KeyCode::Home => {
            state.move_to_line_start();
            RunningAction::Continue
        }
        KeyCode::End => {
            state.move_to_line_end();
            RunningAction::Continue
        }
        KeyCode::Up => {
            state.move_vertical(-1);
            RunningAction::Continue
        }
        KeyCode::Down => {
            state.move_vertical(1);
            RunningAction::Continue
        }
        KeyCode::Char(ch) => {
            state.insert_char(ch);
            RunningAction::Continue
        }
        _ => RunningAction::Continue,
    }
}

fn render_bottom_frame(frame: &mut Frame<'_>, state: &TuiState) {
    let area = frame.area();
    let width = area.width as usize;
    let height = area.height as usize;
    let surface = bottom_surface(state, width, height);

    frame.render_widget(
        Paragraph::new(Text::from(ratatui_lines(&surface.lines, width))),
        area,
    );
    frame.set_cursor_position(Position::new(
        area.x + surface.cursor_col.min(width.saturating_sub(1)) as u16,
        area.y + surface.cursor_row.min(height.saturating_sub(1)) as u16,
    ));
}

fn render_selection_frame(frame: &mut Frame<'_>, state: &TuiState) {
    let area = frame.area();
    let width = area.width as usize;
    let height = area.height as usize;
    let surface = selection_surface(state, width, height);

    frame.render_widget(
        Paragraph::new(Text::from(ratatui_lines(&surface.lines, width))),
        area,
    );
    frame.set_cursor_position(Position::new(area.x, area.y));
}

#[cfg(test)]
fn bottom_lines(state: &TuiState, width: usize, height: usize) -> (Vec<StyledLine>, usize, usize) {
    let surface = bottom_surface(state, width, height);
    (surface.lines, surface.cursor_row, surface.cursor_col)
}

fn bottom_surface(state: &TuiState, width: usize, height: usize) -> BottomSurface {
    let width = width.max(1);
    let height = height.max(1);
    let info_line = state.info_line();
    let info_rows = usize::from(!info_line.is_empty() && height >= 2);
    let queued_lines = if height >= 3 {
        queued_preview_lines(state, width)
    } else {
        Vec::new()
    };
    let queued_rows = queued_lines.len();
    let input_lines = render_input_lines(&state.input, width);
    let live_active = state.live_assistant.is_some()
        || state.live_tool.is_some()
        || state.status.starts_with("Running");
    let input_rows = visible_input_row_count(
        input_lines.len(),
        height,
        info_rows + queued_rows,
        live_active,
    );
    let input_start = input_lines.len().saturating_sub(input_rows);
    let visible_input = input_lines
        .iter()
        .enumerate()
        .skip(input_start)
        .take(input_rows)
        .map(|(index, line)| {
            let prefix = if index == 0 {
                INPUT_PREFIX
            } else {
                INPUT_CONTINUATION_PREFIX
            };
            StyledLine::new(format!("{prefix}{line}"), LineKind::Input)
        })
        .collect::<Vec<_>>();

    let tail_rows = visible_input.len() + queued_rows + info_rows;
    let live_budget = height
        .saturating_sub(tail_rows)
        .saturating_sub(usize::from(tail_rows > 0));
    let mut live_lines = state.render_live_lines(width, live_budget);
    let has_live_separator =
        !live_lines.is_empty() && tail_rows > 0 && live_lines.len() + tail_rows < height;
    let live_rows = live_lines.len();

    let mut lines = Vec::new();
    lines.append(&mut live_lines);
    if has_live_separator {
        lines.push(StyledLine::blank());
    }
    lines.extend(queued_lines);
    let input_row_offset = lines.len();
    lines.extend(visible_input);
    if info_rows > 0 {
        lines.push(StyledLine::new(info_line, LineKind::Hint));
    }

    if lines.len() > height {
        lines = lines.split_off(lines.len() - height);
    }

    while lines.len() < height {
        lines.push(StyledLine::blank());
    }

    let (cursor_row, cursor_col) = cursor_position_for_input(&state.input, state.cursor, width, 0);
    let visible_cursor_row = (cursor_row as usize).saturating_sub(input_start).min(
        input_lines
            .len()
            .saturating_sub(input_start)
            .saturating_sub(1),
    );
    let row = input_row_offset + visible_cursor_row;

    BottomSurface {
        lines,
        cursor_row: row.min(height.saturating_sub(1)),
        cursor_col,
        live_rows,
        input_rows,
        info_rows,
    }
}

fn selection_surface(state: &TuiState, width: usize, height: usize) -> SelectionSurface {
    let width = width.max(1);
    let height = height.max(1);
    let title = state.selection_title.as_deref().unwrap_or("Select");
    let footer_rows = usize::from(height >= 2);
    let header_limit = height.saturating_sub(footer_rows + 1).clamp(1, 4);
    let mut lines = selection_header_lines(title, width, header_limit);
    let body_capacity = height.saturating_sub(lines.len() + footer_rows).max(1);
    let has_preview = width >= SELECTION_PREVIEW_MIN_WIDTH && body_capacity >= 3;
    let max_scroll = state.selection_items.len().saturating_sub(body_capacity);
    let start = state
        .selection_index
        .saturating_sub(body_capacity / 2)
        .min(max_scroll);
    let end = (start + body_capacity).min(state.selection_items.len());
    let selected_row = lines.len() + state.selection_index.saturating_sub(start);
    let (left_width, preview_width) = selection_column_widths(width, has_preview);
    let preview_lines = if has_preview {
        selection_preview_lines(state, preview_width, body_capacity)
    } else {
        Vec::new()
    };

    for item_index in start..end {
        let relative_row = item_index - start;
        let marker = if item_index == state.selection_index {
            "›"
        } else {
            " "
        };
        let kind = if item_index == state.selection_index {
            LineKind::Selection
        } else {
            LineKind::Plain
        };
        let left = format!("{marker} {}", state.selection_items[item_index]);
        let text = if has_preview {
            let preview = preview_lines.get(relative_row).cloned().unwrap_or_default();
            format!(
                "{} │ {}",
                pad_to_display_width(&truncate_to_display_width(&left, left_width), left_width),
                truncate_to_display_width(&preview, preview_width)
            )
        } else {
            left
        };
        lines.push(StyledLine::new(text, kind));
    }

    if footer_rows > 0 {
        lines.push(StyledLine::new(
            "Enter select · Esc cancel · ↑/↓ move",
            LineKind::Hint,
        ));
    }
    while lines.len() < height {
        lines.push(StyledLine::blank());
    }

    if lines.len() > height {
        lines.truncate(height);
    }

    SelectionSurface {
        lines,
        selected_row: selected_row.min(height.saturating_sub(1)),
        has_preview,
    }
}

fn visible_input_row_count(
    rendered_input_rows: usize,
    height: usize,
    reserved_rows: usize,
    live_active: bool,
) -> usize {
    let remaining = height.saturating_sub(reserved_rows).max(1);
    let input_limit = if live_active {
        (height / 3).clamp(1, 5)
    } else {
        remaining
    };
    rendered_input_rows.max(1).min(remaining).min(input_limit)
}

fn queued_preview_lines(state: &TuiState, width: usize) -> Vec<StyledLine> {
    let Some(input) = state.queued_input.as_deref() else {
        return Vec::new();
    };
    if input.trim().is_empty() {
        return Vec::new();
    }

    let prefix = "Queued next: ";
    let preview_width = width.saturating_sub(display_width(prefix)).max(1);
    vec![StyledLine::new(
        format!("{prefix}{}", one_line_preview(input, preview_width)),
        LineKind::Status,
    )]
}

fn one_line_preview(text: &str, width: usize) -> String {
    let compact = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" / ");
    let compact = if compact.is_empty() {
        text.trim()
    } else {
        &compact
    };
    truncate_to_display_width(compact, width)
}

fn selection_header_lines(title: &str, width: usize, max_rows: usize) -> Vec<StyledLine> {
    let mut lines = Vec::new();
    for raw_line in title.lines() {
        let raw_line = raw_line.trim();
        if raw_line.is_empty() {
            continue;
        }

        for segment in wrap_line_by_display_width(raw_line, width) {
            let kind = if lines.is_empty() {
                LineKind::Status
            } else {
                LineKind::Hint
            };
            lines.push(StyledLine::new(segment, kind));
            if lines.len() >= max_rows {
                return lines;
            }
        }
    }

    if lines.is_empty() {
        lines.push(StyledLine::new("Select", LineKind::Status));
    }
    lines
}

fn selection_column_widths(width: usize, has_preview: bool) -> (usize, usize) {
    if !has_preview {
        return (width, 0);
    }

    let preview_width = (width / 3).clamp(24, 38).min(width.saturating_sub(36));
    let left_width = width.saturating_sub(preview_width + 3).max(1);
    (left_width, preview_width)
}

fn selection_preview_lines(state: &TuiState, width: usize, max_rows: usize) -> Vec<String> {
    let mut raw_lines = Vec::new();
    raw_lines.push(format!(
        "Selected {}/{}",
        state.selection_index + 1,
        state.selection_items.len()
    ));
    if let Some(item) = state.selection_items.get(state.selection_index) {
        raw_lines.push(item.clone());
    }
    if let Some(title) = state.selection_title.as_deref() {
        raw_lines.extend(
            title
                .lines()
                .skip(1)
                .take(2)
                .map(|line| line.trim().to_string()),
        );
    }

    let mut lines = Vec::new();
    for raw_line in raw_lines {
        if raw_line.is_empty() {
            continue;
        }
        lines.extend(wrap_line_by_display_width(&raw_line, width));
        if lines.len() >= max_rows {
            break;
        }
    }
    lines.truncate(max_rows);
    lines
}

fn live_tool_tail_budget(max_lines: usize) -> usize {
    max_lines.saturating_sub(1).max(1).min(6)
}

fn ratatui_lines(lines: &[StyledLine], width: usize) -> Vec<Line<'static>> {
    lines
        .iter()
        .map(|line| {
            let clipped = clip_to_width(&line.text, width);
            let text = match line.kind {
                LineKind::Selection | LineKind::User => pad_to_display_width(&clipped, width),
                _ => clipped,
            };
            Line::from(Span::styled(text, style_for_kind(line.kind)))
        })
        .collect()
}

fn style_for_kind(kind: LineKind) -> Style {
    match kind {
        LineKind::Plain | LineKind::Assistant | LineKind::Input => Style::default(),
        LineKind::User => Style::default().fg(Color::White).bg(Color::Rgb(42, 46, 51)),
        LineKind::System => Style::default().fg(Color::DarkGray),
        LineKind::Tool => Style::default().fg(Color::DarkGray),
        LineKind::ToolTitle => Style::default()
            .fg(Color::Gray)
            .add_modifier(Modifier::BOLD),
        LineKind::Selection => Style::default()
            .fg(Color::White)
            .bg(Color::Rgb(67, 72, 78))
            .add_modifier(Modifier::BOLD),
        LineKind::Status => Style::default().fg(Color::Gray),
        LineKind::Hint => Style::default().fg(Color::DarkGray),
    }
}

fn wrap_lines(lines: &[StyledLine], width: usize) -> Vec<StyledLine> {
    let width = width.max(1);
    let mut wrapped = Vec::new();

    for line in lines {
        if line.text.is_empty() {
            wrapped.push(StyledLine::new("", line.kind));
            continue;
        }

        wrapped.extend(
            wrap_line_by_display_width(&line.text, width)
                .into_iter()
                .map(|segment| StyledLine::new(segment, line.kind)),
        );
    }

    if wrapped.is_empty() {
        wrapped.push(StyledLine::blank());
    }

    wrapped
}

fn clip_to_width(text: &str, width: usize) -> String {
    let mut clipped = String::new();
    let mut used = 0;
    for ch in text.chars() {
        let ch_width = display_width_char(ch);
        if !clipped.is_empty() && used + ch_width > width {
            break;
        }
        clipped.push(ch);
        used += ch_width;
        if used >= width {
            break;
        }
    }
    clipped
}

fn truncate_to_display_width(text: &str, width: usize) -> String {
    let width = width.max(1);
    if display_width(text) <= width {
        return text.to_string();
    }
    if width <= 3 {
        return clip_to_width(text, width);
    }

    let mut clipped = clip_to_width(text, width - 3);
    clipped.push_str("...");
    clipped
}

fn char_to_byte_index(text: &str, char_index: usize) -> usize {
    if char_index == 0 {
        return 0;
    }
    text.char_indices()
        .nth(char_index)
        .map(|(index, _)| index)
        .unwrap_or(text.len())
}

fn row_col_for_cursor(chars: &[char], cursor: usize) -> (usize, usize) {
    let mut row = 0;
    let mut col = 0;
    for ch in chars.iter().take(cursor) {
        if *ch == '\n' {
            row += 1;
            col = 0;
        } else {
            col += display_width_char(*ch);
        }
    }
    (row, col)
}

fn cursor_for_row_col(chars: &[char], target_row: usize, target_col: usize) -> usize {
    let mut row = 0;
    let mut col = 0;

    for (index, ch) in chars.iter().enumerate() {
        if row == target_row && col == target_col {
            return index;
        }

        if *ch == '\n' {
            if row == target_row {
                return index;
            }
            row += 1;
            col = 0;
        } else {
            col += display_width_char(*ch);
        }
    }

    chars.len()
}

fn display_width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

fn display_width_char(ch: char) -> usize {
    UnicodeWidthChar::width(ch).unwrap_or(0)
}

fn should_handle_key_event(key: KeyEvent) -> bool {
    matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat)
}

fn initial_viewport_size() -> (u16, u16) {
    crossterm::terminal::size()
        .map(|(columns, rows)| (columns, inline_viewport_height_for_rows(rows)))
        .unwrap_or((0, DEFAULT_INLINE_VIEWPORT_HEIGHT))
}

fn inline_viewport_height_for_rows(rows: u16) -> u16 {
    rows.saturating_mul(3)
        .saturating_div(5)
        .clamp(MIN_INLINE_VIEWPORT_HEIGHT, MAX_INLINE_VIEWPORT_HEIGHT)
}

fn selection_page_size(viewport_height: u16) -> usize {
    let height = if viewport_height == 0 {
        DEFAULT_INLINE_VIEWPORT_HEIGHT
    } else {
        viewport_height
    };
    height.saturating_sub(3).max(1) as usize
}

fn format_elapsed(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    if seconds < 60 {
        format!("{seconds}s")
    } else {
        let minutes = seconds / 60;
        let remainder = seconds % 60;
        format!("{minutes}m {remainder}s")
    }
}

fn current_time_label() -> String {
    let now = OffsetDateTime::now_utc();
    let local = UtcOffset::current_local_offset()
        .ok()
        .map(|offset| now.to_offset(offset))
        .unwrap_or(now);
    let year = local.year() % 100;
    let month = u8::from(local.month());
    let day = local.day();
    let hour = local.hour();
    let minute = local.minute();
    let second = local.second();
    format!("{year:02}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02}")
}

fn format_token_count(total: u64) -> String {
    if total >= 1_000_000 {
        format!("{:.1}m", total as f64 / 1_000_000.0)
    } else if total >= 1_000 {
        format!("{:.1}k", total as f64 / 1_000.0)
    } else {
        total.to_string()
    }
}

fn pad_to_display_width(text: &str, width: usize) -> String {
    let current = display_width(text);
    if current >= width {
        text.to_string()
    } else {
        format!("{text}{}", " ".repeat(width - current))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn running_keys_map_to_abort_and_quit_actions() {
        let mut state = TuiState::default();

        let abort =
            handle_running_key_state(&mut state, KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(matches!(abort, RunningAction::Abort));

        let quit = handle_running_key_state(
            &mut state,
            KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
        );
        assert!(matches!(quit, RunningAction::Quit));

        let cont =
            handle_running_key_state(&mut state, KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert!(matches!(cont, RunningAction::Continue));
    }

    #[test]
    fn prompt_reset_keeps_existing_status() {
        let mut state = TuiState::default();
        state.set_status("Provider mismatch, /fork recommended");
        state.input = "pending".to_string();
        state.cursor = 7;

        state.input.clear();
        state.cursor = 0;

        assert_eq!(state.status, "Provider mismatch, /fork recommended");
        assert!(state.input.is_empty());
        assert_eq!(state.cursor, 0);
    }

    #[test]
    fn system_notes_split_multiline_text_into_lines() {
        let mut state = TuiState::default();
        state.push_system_note("first line\nsecond line");

        assert_eq!(
            block_texts(state.committed_blocks.last().unwrap()),
            vec!["· first line".to_string(), "· second line".to_string()]
        );
    }

    #[test]
    fn format_message_splits_multiline_tool_output() {
        let message = AgentMessage::ToolResult(agent_model::LlmMessage {
            role: agent_model::LlmRole::Tool,
            parts: vec![agent_model::MessagePart::ToolResult(
                agent_model::ToolResultPart {
                    call_id: "call-1".to_string(),
                    content: "line one\nline two".to_string(),
                    is_error: false,
                },
            )],
        });

        assert_eq!(
            format_message(&message)
                .into_iter()
                .map(|line| line.text)
                .collect::<Vec<_>>(),
            vec![
                "· tool result call-1".to_string(),
                "  line one".to_string(),
                "  line two".to_string(),
            ]
        );
    }

    #[test]
    fn assistant_messages_use_body_line_kind() {
        let message = AgentMessage::Assistant(agent_model::LlmMessage {
            role: agent_model::LlmRole::Assistant,
            parts: vec![agent_model::MessagePart::Text(agent_model::TextPart {
                text: "readable answer".to_string(),
            })],
        });

        let lines = format_message(&message);
        assert_eq!(lines[0].kind, LineKind::Assistant);
        assert_eq!(lines[0].text, "readable answer");
    }

    #[test]
    fn long_tool_outputs_are_compacted_around_the_tail() {
        let content = (1..=12)
            .map(|index| format!("line {index}"))
            .collect::<Vec<_>>()
            .join("\n");
        let message = AgentMessage::ToolResult(agent_model::LlmMessage {
            role: agent_model::LlmRole::Tool,
            parts: vec![agent_model::MessagePart::ToolResult(
                agent_model::ToolResultPart {
                    call_id: "call-1".to_string(),
                    content,
                    is_error: false,
                },
            )],
        });

        let texts = format_message(&message)
            .into_iter()
            .map(|line| line.text)
            .collect::<Vec<_>>();

        assert_eq!(texts[0], "· tool result call-1");
        assert!(texts.contains(&"  ... 4 earlier lines omitted".to_string()));
        assert!(!texts.contains(&"  line 1".to_string()));
        assert!(texts.contains(&"  line 12".to_string()));
    }

    #[test]
    fn live_tool_args_do_not_merge_into_the_tool_title() {
        let mut block = RenderBlock {
            lines: vec![StyledLine::new("· tool bash (call-1)", LineKind::ToolTitle)],
            kind: BlockKind::Tool,
        };

        append_text_to_block(&mut block, "  ", "{\"cmd\":\"ls\"}");

        assert_eq!(block.lines[0].text, "· tool bash (call-1)");
        assert_eq!(block.lines[0].kind, LineKind::ToolTitle);
        assert_eq!(block.lines[1].text, "  {\"cmd\":\"ls\"}");
        assert_eq!(block.lines[1].kind, LineKind::Tool);
    }

    #[test]
    fn input_wrapping_uses_display_width_for_wide_characters() {
        let lines = render_input_lines("안녕하세요", 8);
        assert_eq!(lines, vec!["안녕하".to_string(), "세요".to_string()]);
    }

    #[test]
    fn insert_text_preserves_multiline_paste() {
        let mut state = TuiState::default();
        state.insert_text("first\nsecond");

        assert_eq!(state.input, "first\nsecond");
        assert_eq!(state.cursor, "first\nsecond".chars().count());
    }

    #[test]
    fn cursor_position_accounts_for_wider_input_prefix() {
        let (row, col) = cursor_position_for_input("hello", 5, 20, 6);
        assert_eq!(row, 6);
        assert_eq!(col, INPUT_PREFIX_WIDTH + 5);
    }

    #[test]
    fn token_count_is_compactly_formatted() {
        assert_eq!(format_token_count(999), "999");
        assert_eq!(format_token_count(1_500), "1.5k");
        assert_eq!(format_token_count(2_500_000), "2.5m");
    }

    #[test]
    fn inline_viewport_height_tracks_terminal_rows_with_bounds() {
        assert_eq!(
            inline_viewport_height_for_rows(12),
            MIN_INLINE_VIEWPORT_HEIGHT
        );
        assert_eq!(
            inline_viewport_height_for_rows(24),
            DEFAULT_INLINE_VIEWPORT_HEIGHT
        );
        assert_eq!(
            inline_viewport_height_for_rows(80),
            MAX_INLINE_VIEWPORT_HEIGHT
        );
    }

    #[test]
    fn user_messages_use_distinct_line_kind() {
        let message = AgentMessage::User(agent_model::LlmMessage {
            role: agent_model::LlmRole::User,
            parts: vec![agent_model::MessagePart::Text(agent_model::TextPart {
                text: "hello".to_string(),
            })],
        });

        let lines = format_message(&message);
        assert_eq!(lines[0].kind, LineKind::User);
        assert_eq!(lines[0].text, "▌  hello");
    }

    #[test]
    fn running_enter_stores_queued_prompt_preview() {
        let mut state = TuiState {
            input: "next\nprompt".to_string(),
            cursor: "next\nprompt".chars().count(),
            ..Default::default()
        };

        let action = handle_running_key_state(
            &mut state,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
        );

        assert!(matches!(action, RunningAction::QueueSubmit(text) if text == "next\nprompt"));
        assert_eq!(state.queued_input.as_deref(), Some("next\nprompt"));
        let surface = bottom_surface(&state, 40, 5);
        assert!(surface
            .lines
            .iter()
            .any(|line| line.text == "Queued next: next / prompt"));
    }

    #[test]
    fn bottom_frame_keeps_prompt_adjacent_to_transcript_when_idle() {
        let state = TuiState {
            input: "hello".to_string(),
            cursor: 5,
            ..Default::default()
        };

        let (lines, cursor_row, cursor_col) = bottom_lines(&state, 20, 4);
        let texts = lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>();

        assert_eq!(texts[0], "> hello");
        assert!(texts[1].contains("usage n/a"));
        assert_eq!(cursor_row, 0);
        assert_eq!(cursor_col, INPUT_PREFIX_WIDTH + 5);
    }

    #[test]
    fn bottom_frame_uses_available_space_for_live_assistant_output() {
        let mut state = TuiState::default();
        state.set_status("Running");
        state.apply_event(AgentEvent::MessageStart {
            role: agent_model::LlmRole::Assistant,
        });
        state.apply_event(AgentEvent::TextDelta(
            "one\ntwo\nthree\nfour\nfive".to_string(),
        ));

        let (lines, _, _) = bottom_lines(&state, 30, 10);
        let texts = lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>();

        assert!(texts.contains(&"one"));
        assert!(texts.contains(&"two"));
        assert!(texts.contains(&"three"));
        assert!(texts.contains(&"four"));
        assert!(texts.contains(&"five"));
        assert!(!texts.contains(&"Reply preview"));
    }

    #[test]
    fn bottom_surface_keeps_live_tail_visible_when_input_wraps() {
        let mut state = TuiState {
            input: "first line\nsecond line\nthird line\nfourth line".to_string(),
            cursor: "first line\nsecond line\nthird line\nfourth line"
                .chars()
                .count(),
            ..Default::default()
        };
        state.set_status("Running");
        state.apply_event(AgentEvent::MessageStart {
            role: agent_model::LlmRole::Assistant,
        });
        state.apply_event(AgentEvent::TextDelta(
            "one\ntwo\nthree\nfour\nfive".to_string(),
        ));

        let surface = bottom_surface(&state, 40, 8);
        let texts = surface
            .lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>();

        assert!(surface.live_rows > 0);
        assert!(surface.input_rows <= 2);
        assert!(texts.contains(&"five"));
        assert!(texts
            .iter()
            .any(|line| line.starts_with("> ") || line.starts_with("  ")));
    }

    #[test]
    fn test_backend_keeps_live_output_above_prompt_and_footer() {
        use ratatui::backend::TestBackend;

        let mut state = TuiState {
            input: "queued?".to_string(),
            cursor: 7,
            ..Default::default()
        };
        state.set_status("Running");
        state.apply_event(AgentEvent::MessageStart {
            role: agent_model::LlmRole::Assistant,
        });
        state.apply_event(AgentEvent::TextDelta("alpha\nbeta\ngamma".to_string()));

        let backend = TestBackend::new(72, 6);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render_frame(frame, &state)).unwrap();
        let lines = buffer_lines(terminal.backend().buffer());
        let prompt_row = lines
            .iter()
            .position(|line| line.starts_with("> queued?"))
            .unwrap();
        let gamma_row = lines
            .iter()
            .position(|line| line.trim() == "gamma")
            .unwrap();

        assert!(gamma_row < prompt_row);
        assert!(prompt_row < lines.len() - 1);
        assert!(lines.last().unwrap().contains("usage n/a"));
    }

    #[test]
    fn selection_surface_shows_preview_for_wide_viewports() {
        let state = TuiState {
            selection_title: Some("Select session (= match, ! mismatch, ? unknown)".to_string()),
            selection_items: vec![
                "= 26-05-22 10:00 openai/gpt session-a".to_string(),
                "! 26-05-22 11:00 anthropic/claude session-b".to_string(),
            ],
            selection_index: 1,
            ..Default::default()
        };

        let surface = selection_surface(&state, 90, 8);
        let texts = surface
            .lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>();

        assert!(surface.has_preview);
        assert!(texts.iter().any(|line| line.contains("│ Selected 2/2")));
        assert!(texts[surface.selected_row].contains("session-b"));
    }

    #[test]
    fn selection_surface_keeps_mismatch_details_readable() {
        let state = TuiState {
            selection_title: Some(
                "Session provider differs\nruntime=openai/gpt-4.1\nsession=anthropic/claude\nmodel: runtime=gpt-4.1 session=claude".to_string(),
            ),
            selection_items: vec![
                "Fork with current provider (openai) [recommended]".to_string(),
                "Open existing session as-is [advanced]".to_string(),
                "Cancel".to_string(),
            ],
            selection_index: 0,
            ..Default::default()
        };

        let surface = selection_surface(&state, 82, 9);
        let texts = surface
            .lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>();

        assert_eq!(texts[0], "Session provider differs");
        assert!(texts
            .iter()
            .any(|line| line.contains("runtime=openai/gpt-4.1")));
        assert!(texts[surface.selected_row].contains("Fork with current provider"));
    }

    #[test]
    fn ratatui_inline_viewport_preserves_committed_scrollback() {
        use ratatui::backend::TestBackend;

        let backend = TestBackend::new(24, 4);
        let mut terminal = Terminal::with_options(
            backend,
            TerminalOptions {
                viewport: Viewport::Inline(2),
            },
        )
        .unwrap();

        terminal
            .draw(|frame| {
                Paragraph::new("> prompt\nstatus").render(frame.area(), frame.buffer_mut());
            })
            .unwrap();

        terminal
            .insert_before(3, |buf| {
                Paragraph::new("first\nsecond\nthird").render(buf.area, buf);
            })
            .unwrap();
        terminal
            .insert_before(2, |buf| {
                Paragraph::new("fourth\nfifth").render(buf.area, buf);
            })
            .unwrap();
        terminal
            .draw(|frame| {
                Paragraph::new("> prompt\nstatus").render(frame.area(), frame.buffer_mut());
            })
            .unwrap();

        terminal.backend().assert_scrollback_lines([
            "first                   ",
            "second                  ",
            "third                   ",
        ]);
        terminal.backend().assert_buffer_lines([
            "fourth                  ",
            "fifth                   ",
            "> prompt                ",
            "status                  ",
        ]);
    }

    fn block_texts(block: &RenderBlock) -> Vec<String> {
        block.lines.iter().map(|line| line.text.clone()).collect()
    }

    fn buffer_lines(buffer: &ratatui::buffer::Buffer) -> Vec<String> {
        (buffer.area.y..buffer.area.y + buffer.area.height)
            .map(|y| {
                (buffer.area.x..buffer.area.x + buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect()
    }
}

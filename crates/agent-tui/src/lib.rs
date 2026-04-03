use std::io::{Stdout, Write, stdout};
use std::time::{Duration, Instant};

use anyhow::Context;
use agent_model::Usage;
use crossterm::cursor::{MoveTo, Show};
use crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers, KeyboardEnhancementFlags, MouseEvent, MouseEventKind,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::style::{Color, Print, ResetColor, SetBackgroundColor, SetForegroundColor};
use crossterm::terminal::{self, Clear, ClearType, disable_raw_mode, enable_raw_mode};
use crossterm::{execute, queue};
use time::OffsetDateTime;
use time::UtcOffset;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use agent_core::{AgentEvent, AgentMessage};
use bottom_pane::{cursor_position_for_input, render_input_lines, wrap_line_by_display_width};
use conversation::{append_text_to_block, classify_block_from_message, format_message, message_lines};

mod bottom_pane;
mod conversation;

const INPUT_PREFIX: &str = "> ";
const INPUT_CONTINUATION_PREFIX: &str = "  ";
const INPUT_PREFIX_WIDTH: usize = 2;

const USER_PREFIX: &str = "▌  ";

pub struct TuiApp {
    terminal: Terminal,
    state: TuiState,
    last_frame: Vec<StyledLine>,
}

impl TuiApp {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            terminal: Terminal::new(),
            state: TuiState::default(),
            last_frame: Vec::new(),
        })
    }

    pub fn enter(&mut self) -> anyhow::Result<()> {
        self.terminal.enter()?;
        self.last_frame.clear();
        self.render()
    }

    pub fn leave(&mut self) -> anyhow::Result<()> {
        self.terminal.leave()
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
        self.state.selection_title = Some(title.into());
        self.state.selection_items = items.to_vec();
        self.state.selection_index = initial_index.min(self.state.selection_items.len().saturating_sub(1));

        loop {
            self.render()?;
            if !event::poll(Duration::from_millis(50)).context("failed to poll terminal events")? {
                continue;
            }

        match event::read().context("failed to read terminal event")? {
            Event::Key(key) if should_handle_key_event(key) => match key.code {
                KeyCode::Esc => {
                    self.state.selection_title = None;
                    self.state.selection_items.clear();
                    return Ok(None);
                    }
                    KeyCode::Enter => {
                        let index = self
                            .state
                            .selection_index
                            .min(self.state.selection_items.len().saturating_sub(1));
                        self.state.selection_title = None;
                        self.state.selection_items.clear();
                        return Ok(Some(index));
                    }
                    KeyCode::Up => {
                        self.state.selection_index = self.state.selection_index.saturating_sub(1);
                    }
                    KeyCode::Down => {
                        let max = self.state.selection_items.len().saturating_sub(1);
                        self.state.selection_index = (self.state.selection_index + 1).min(max);
                    }
                    _ => {}
                },
            Event::Key(_) => {}
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
            Event::Key(_) => Ok(RunningAction::Continue),
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

    pub fn poll_prompt_action(&mut self, timeout: Duration) -> anyhow::Result<Option<PromptAction>> {
        self.render()?;
        if !event::poll(timeout).context("failed to poll terminal events")? {
            return Ok(None);
        }

        match event::read().context("failed to read terminal event")? {
            Event::Key(key) if should_handle_key_event(key) => Ok(self.handle_key(key)),
            Event::Key(_) => Ok(Some(PromptAction::Continue)),
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

    fn handle_key(&mut self, key: KeyEvent) -> Option<PromptAction> {
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => Some(PromptAction::Quit),
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
                if key.modifiers.contains(KeyModifiers::CONTROL) {
                    self.state.scroll_up(1);
                } else {
                    self.state.move_vertical(-1);
                }
                Some(PromptAction::Continue)
            }
            KeyCode::Down => {
                if key.modifiers.contains(KeyModifiers::CONTROL) {
                    self.state.scroll_down(1);
                } else {
                    self.state.move_vertical(1);
                }
                Some(PromptAction::Continue)
            }
            KeyCode::PageUp => {
                self.state.scroll_up(self.state.page_scroll_amount());
                Some(PromptAction::Continue)
            }
            KeyCode::PageDown => {
                self.state.scroll_down(self.state.page_scroll_amount());
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
        match key.code {
            KeyCode::Esc => RunningAction::Abort,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => RunningAction::Quit,
            KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.state.insert_char('\n');
                RunningAction::Continue
            }
            KeyCode::Enter
                if key.modifiers.contains(KeyModifiers::SHIFT)
                    || key.modifiers.contains(KeyModifiers::ALT) =>
            {
                self.state.insert_char('\n');
                RunningAction::Continue
            }
            KeyCode::Enter => {
                let text = std::mem::take(&mut self.state.input);
                self.state.cursor = 0;
                if text.trim().is_empty() {
                    RunningAction::Continue
                } else {
                    RunningAction::QueueSubmit(text)
                }
            }
            KeyCode::Backspace => {
                self.state.backspace();
                RunningAction::Continue
            }
            KeyCode::Delete => {
                self.state.delete();
                RunningAction::Continue
            }
            KeyCode::Left => {
                self.state.move_left();
                RunningAction::Continue
            }
            KeyCode::Right => {
                self.state.move_right();
                RunningAction::Continue
            }
            KeyCode::Home => {
                self.state.move_to_line_start();
                RunningAction::Continue
            }
            KeyCode::End => {
                self.state.move_to_line_end();
                RunningAction::Continue
            }
            KeyCode::Up => {
                self.state.scroll_up(1);
                RunningAction::Continue
            }
            KeyCode::Down => {
                self.state.scroll_down(1);
                RunningAction::Continue
            }
            KeyCode::PageUp => {
                self.state.scroll_up(self.state.page_scroll_amount());
                RunningAction::Continue
            }
            KeyCode::PageDown => {
                self.state.scroll_down(self.state.page_scroll_amount());
                RunningAction::Continue
            }
            KeyCode::Char(ch) => {
                self.state.insert_char(ch);
                RunningAction::Continue
            }
            _ => RunningAction::Continue,
        }
    }

    fn handle_mouse(&mut self, mouse: MouseEvent) {
        match mouse.kind {
            MouseEventKind::ScrollUp => self.state.scroll_up(3),
            MouseEventKind::ScrollDown => self.state.scroll_down(3),
            _ => {}
        }
    }

    fn handle_mouse_in_selection(&mut self, mouse: MouseEvent) {
        match mouse.kind {
            MouseEventKind::ScrollUp => {
                self.state.selection_index = self.state.selection_index.saturating_sub(1);
            }
            MouseEventKind::ScrollDown => {
                let max = self.state.selection_items.len().saturating_sub(1);
                self.state.selection_index = (self.state.selection_index + 1).min(max);
            }
            _ => self.handle_mouse(mouse),
        }
    }

    fn render(&mut self) -> anyhow::Result<()> {
        let (width, height) = terminal::size().context("failed to get terminal size")?;
        self.state.viewport_width = width;
        self.state.viewport_height = height;

        let raw_lines = self.state.render_lines();
        let wrapped_lines = wrap_lines(&raw_lines, width as usize);

        let input_lines = render_input_lines(&self.state.input, width as usize);
        let editor_height = input_lines.len().max(1) as u16;
        let hint_rows = 1u16;
        let bottom_divider_row = height.saturating_sub(hint_rows + 1);
        let input_start_row = bottom_divider_row.saturating_sub(editor_height);
        let top_divider_row = input_start_row.saturating_sub(1);
        let status_text = self.state.status_line();
        let status_row = if !status_text.is_empty() && top_divider_row > 1 {
            Some(top_divider_row - 2)
        } else if !status_text.is_empty() && top_divider_row > 0 {
            Some(top_divider_row - 1)
        } else {
            None
        };
        let hint_row = bottom_divider_row.saturating_add(1);
        let log_height = status_row.unwrap_or(top_divider_row) as usize;
        self.state.update_rendered_lines(wrapped_lines, log_height);
        let visible_lines = self.state.visible_log_lines();

        let mut frame = vec![StyledLine::blank(); height as usize];
        for (index, line) in visible_lines.iter().enumerate() {
            frame[index] = line.with_text(clip_to_width(&line.text, width as usize));
        }

        let hint_text = self.state.hint_line();
        if let Some(status_row) = status_row {
            if status_row < height {
                frame[status_row as usize] =
                    StyledLine::new(clip_to_width(&status_text, width as usize), LineKind::Status);
            }
        }
        if top_divider_row < height {
            frame[top_divider_row as usize] =
                StyledLine::new("─".repeat(width as usize), LineKind::Divider);
        }
        if bottom_divider_row < height {
            frame[bottom_divider_row as usize] =
                StyledLine::new("─".repeat(width as usize), LineKind::Divider);
        }
        if hint_row < height {
            frame[hint_row as usize] =
                StyledLine::new(clip_to_width(&hint_text, width as usize), LineKind::Hint);
        }

        for (offset, line) in input_lines.iter().enumerate() {
            let row = input_start_row as usize + offset;
            if row < frame.len() {
                let prefix = if offset == 0 {
                    INPUT_PREFIX
                } else {
                    INPUT_CONTINUATION_PREFIX
                };
                frame[row] = StyledLine::new(
                    clip_to_width(&format!("{prefix}{line}"), width as usize),
                    LineKind::Input,
                );
            }
        }

        if !self.state.selection_items.is_empty() {
            overlay_selection(
                &mut frame,
                width as usize,
                height as usize,
                self.state.selection_title.as_deref().unwrap_or("Select"),
                &self.state.selection_items,
                self.state.selection_index,
            );
        }

        let (cursor_row, cursor_col) = cursor_position_for_input(
            &self.state.input,
            self.state.cursor,
            self.state.viewport_width as usize,
            input_start_row,
        );
        self.draw_frame(&frame, cursor_row, cursor_col)?;
        self.terminal.stdout.flush()?;
        Ok(())
    }

    fn draw_frame(
        &mut self,
        frame: &[StyledLine],
        cursor_row: u16,
        cursor_col: usize,
    ) -> anyhow::Result<()> {
        if self.last_frame.len() != frame.len() {
            queue!(self.terminal.stdout, MoveTo(0, 0), Clear(ClearType::All))?;
            self.last_frame = vec![StyledLine::blank(); frame.len()];
        }

        for (row, line) in frame.iter().enumerate() {
            let changed = self
                .last_frame
                .get(row)
                .map(|prev| prev != line)
                .unwrap_or(true);
            if changed {
                let clipped = clip_to_width(&line.text, self.state.viewport_width as usize);
                let trailing_spaces = " ".repeat(
                    self.state
                        .viewport_width
                        .saturating_sub(display_width(&clipped) as u16) as usize,
                );
                queue!(self.terminal.stdout, MoveTo(0, row as u16))?;
                apply_style(&mut self.terminal.stdout, line.kind)?;
                queue!(
                    self.terminal.stdout,
                    Print(&clipped),
                    ResetColor,
                    Print(trailing_spaces)
                )?;
            }
        }

        queue!(self.terminal.stdout, Show, MoveTo(cursor_col as u16, cursor_row))?;
        self.last_frame = frame.to_vec();
        Ok(())
    }
}

#[derive(Default)]
struct TuiState {
    committed_blocks: Vec<RenderBlock>,
    live_assistant: Option<RenderBlock>,
    live_tool: Option<RenderBlock>,
    input: String,
    cursor: usize,
    status: String,
    status_since: Option<Instant>,
    footer_path: String,
    footer_model: String,
    latest_usage: Option<Usage>,
    viewport_width: u16,
    viewport_height: u16,
    scroll_top: usize,
    follow_output: bool,
    unseen_output_lines: usize,
    rendered_lines: Vec<StyledLine>,
    rendered_log_height: usize,
    frozen_lines: Option<Vec<StyledLine>>,
    selection_title: Option<String>,
    selection_items: Vec<String>,
    selection_index: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LineKind {
    Plain,
    User,
    System,
    Tool,
    Selection,
    Divider,
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

    fn with_text(&self, text: impl Into<String>) -> Self {
        Self::new(text, self.kind)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn running_keys_map_to_abort_and_quit_actions() {
        let mut app = TuiApp::new().unwrap();

        let abort = app.handle_running_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(matches!(abort, RunningAction::Abort));

        let quit = app.handle_running_key(KeyEvent::new(
            KeyCode::Char('c'),
            KeyModifiers::CONTROL,
        ));
        assert!(matches!(quit, RunningAction::Quit));

        let cont = app.handle_running_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert!(matches!(cont, RunningAction::Continue));
    }

    #[test]
    fn prompt_reset_keeps_existing_status() {
        let mut app = TuiApp::new().unwrap();
        app.set_status("Provider mismatch, /fork recommended");
        app.state.input = "pending".to_string();
        app.state.cursor = 7;

        app.state.input.clear();
        app.state.cursor = 0;

        assert_eq!(app.state.status, "Provider mismatch, /fork recommended");
        assert!(app.state.input.is_empty());
        assert_eq!(app.state.cursor, 0);
    }

    #[test]
    fn system_notes_split_multiline_text_into_lines() {
        let mut app = TuiApp::new().unwrap();
        app.push_system_note("first line\nsecond line");

        assert_eq!(
            block_texts(app.state.committed_blocks.last().unwrap()),
            vec![
                "· first line".to_string(),
                "· second line".to_string(),
            ]
        );
    }

    #[test]
    fn format_message_splits_multiline_tool_output() {
        let message = AgentMessage::ToolResult(agent_model::LlmMessage {
            role: agent_model::LlmRole::Tool,
            parts: vec![agent_model::MessagePart::ToolResult(agent_model::ToolResultPart {
                call_id: "call-1".to_string(),
                content: "line one\nline two".to_string(),
                is_error: false,
            })],
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
    fn input_wrapping_uses_display_width_for_wide_characters() {
        let lines = render_input_lines("안녕하세요", 8);
        assert_eq!(
            lines,
            vec!["안녕하".to_string(), "세요".to_string()]
        );
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
    fn scrolling_up_stops_following_output() {
        let mut state = TuiState {
            follow_output: true,
            rendered_lines: (0..10)
                .map(|idx| StyledLine::new(format!("line {idx}"), LineKind::Plain))
                .collect(),
            rendered_log_height: 4,
            ..Default::default()
        };

        state.scroll_up(3);
        state.push_system_note("new output");

        assert_eq!(state.scroll_top, 3);
        assert!(!state.follow_output);
        assert_eq!(state.unseen_output_lines, 0);
        assert_eq!(state.frozen_lines.as_ref().unwrap().len(), 10);
    }

    #[test]
    fn visible_log_lines_stay_anchored_when_not_following_output() {
        let state = TuiState {
            scroll_top: 2,
            follow_output: false,
            rendered_log_height: 3,
            frozen_lines: Some(
                (0..10)
                    .map(|idx| StyledLine::new(format!("line {idx}"), LineKind::Plain))
                    .collect(),
            ),
            ..Default::default()
        };

        let visible = state.visible_log_lines();
        let texts = visible.iter().map(|line| line.text.as_str()).collect::<Vec<_>>();
        assert_eq!(texts, vec!["line 2", "line 3", "line 4"]);
    }

    #[test]
    fn token_count_is_compactly_formatted() {
        assert_eq!(format_token_count(999), "999");
        assert_eq!(format_token_count(1_500), "1.5k");
        assert_eq!(format_token_count(2_500_000), "2.5m");
    }

    #[test]
    fn sync_scroll_tracks_unseen_output_when_viewport_is_frozen() {
        let mut state = TuiState {
            scroll_top: 2,
            follow_output: false,
            rendered_log_height: 3,
            rendered_lines: (0..5)
                .map(|idx| StyledLine::new(format!("line {idx}"), LineKind::Plain))
                .collect(),
            frozen_lines: Some(
                (0..5)
                    .map(|idx| StyledLine::new(format!("line {idx}"), LineKind::Plain))
                    .collect(),
            ),
            ..Default::default()
        };

        state.update_rendered_lines(
            (0..7)
                .map(|idx| StyledLine::new(format!("line {idx}"), LineKind::Plain))
                .collect(),
            3,
        );

        assert_eq!(state.scroll_top, 2);
        assert_eq!(state.unseen_output_lines, 2);
        assert!(!state.follow_output);
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

    fn block_texts(block: &RenderBlock) -> Vec<String> {
        block.lines.iter().map(|line| line.text.clone()).collect()
    }

}

struct Terminal {
    stdout: Stdout,
    entered: bool,
}

impl Terminal {
    fn new() -> Self {
        Self {
            stdout: stdout(),
            entered: false,
        }
    }

    fn enter(&mut self) -> anyhow::Result<()> {
        if self.entered {
            return Ok(());
        }

        enable_raw_mode().context("failed to enable raw mode")?;
        execute!(
            self.stdout,
            EnableBracketedPaste,
            PushKeyboardEnhancementFlags(
                KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                    | KeyboardEnhancementFlags::REPORT_EVENT_TYPES
                    | KeyboardEnhancementFlags::REPORT_ALTERNATE_KEYS
            )
        )
        .ok();
        self.entered = true;
        Ok(())
    }

    fn leave(&mut self) -> anyhow::Result<()> {
        if !self.entered {
            return Ok(());
        }

        execute!(
            self.stdout,
            Show,
            DisableBracketedPaste,
            PopKeyboardEnhancementFlags
        )
        .ok();
        disable_raw_mode().context("failed to disable raw mode")?;
        self.entered = false;
        Ok(())
    }
}

impl Drop for Terminal {
    fn drop(&mut self) {
        let _ = self.leave();
    }
}

#[derive(Clone)]
struct RenderBlock {
    lines: Vec<StyledLine>,
    kind: BlockKind,
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

    fn render_lines(&self) -> Vec<StyledLine> {
        let mut lines = Vec::new();
        let mut has_previous_block = false;

        for block in &self.committed_blocks {
            if has_previous_block && block.kind != BlockKind::System {
                lines.push(StyledLine::blank());
            }
            lines.extend(block.lines.iter().cloned());
            has_previous_block = true;
        }

        if let Some(block) = &self.live_assistant {
            if has_previous_block && block.kind != BlockKind::System {
                lines.push(StyledLine::blank());
            }
            lines.extend(block.lines.iter().cloned());
            has_previous_block = true;
        }

        if let Some(block) = &self.live_tool {
            if has_previous_block && block.kind != BlockKind::System {
                lines.push(StyledLine::blank());
            }
            lines.extend(block.lines.iter().cloned());
        }
        lines
    }

    fn visible_log_lines(&self) -> &[StyledLine] {
        let lines = self.active_log_lines();
        if self.rendered_log_height == 0 {
            return &[];
        }
        let max_start = lines.len().saturating_sub(self.rendered_log_height);
        let start = if self.follow_output {
            max_start
        } else {
            self.scroll_top.min(max_start)
        };
        let end = (start + self.rendered_log_height).min(lines.len());
        &lines[start..end]
    }

    fn page_scroll_amount(&self) -> usize {
        self.viewport_height.saturating_sub(3) as usize
    }

    fn scroll_up(&mut self, amount: usize) {
        if self.follow_output {
            self.scroll_top = self
                .rendered_lines
                .len()
                .saturating_sub(self.rendered_log_height);
            self.frozen_lines = Some(self.rendered_lines.clone());
        }
        self.follow_output = false;
        self.scroll_top = self.scroll_top.saturating_sub(amount);
    }

    fn scroll_down(&mut self, amount: usize) {
        if self.follow_output {
            return;
        }

        let max_start = self
            .active_log_lines()
            .len()
            .saturating_sub(self.rendered_log_height);
        let next = self.scroll_top.saturating_add(amount);
        if next >= max_start {
            self.follow_output = true;
            self.scroll_top = self
                .rendered_lines
                .len()
                .saturating_sub(self.rendered_log_height);
            self.unseen_output_lines = 0;
            self.frozen_lines = None;
        } else {
            self.scroll_top = next;
        }
    }

    fn update_rendered_lines(&mut self, log_lines: Vec<StyledLine>, log_height: usize) {
        let previous_rendered_len = self.rendered_lines.len();
        self.rendered_lines = log_lines;
        self.rendered_log_height = log_height;

        let max_start = self.rendered_lines.len().saturating_sub(log_height);
        if self.follow_output {
            self.scroll_top = max_start;
            self.unseen_output_lines = 0;
        } else {
            if self.rendered_lines.len() > previous_rendered_len {
                self.unseen_output_lines = self
                    .unseen_output_lines
                    .saturating_add(self.rendered_lines.len() - previous_rendered_len);
            }
            let frozen_max_start = self
                .frozen_lines
                .as_ref()
                .map(|lines| lines.len().saturating_sub(log_height))
                .unwrap_or(0);
            self.scroll_top = self.scroll_top.min(frozen_max_start);
        }
    }

    fn status_line(&self) -> String {
        let mut parts = Vec::new();
        if !self.status.is_empty() && self.status != "Ready" {
            if self.status.starts_with("Running") {
                parts.push(self.status.clone());
                if let Some(status_since) = self.status_since {
                    parts.push(format_elapsed(status_since.elapsed()));
                }
                parts.push("Esc abort".to_string());
            } else {
                parts.push(self.status.clone());
            }
        }
        if !self.follow_output && self.unseen_output_lines > 0 {
            parts.push(format!(
                "{} new line{} below",
                self.unseen_output_lines,
                if self.unseen_output_lines == 1 { "" } else { "s" }
            ));
        }
        parts.join(" · ")
    }

    fn hint_line(&self) -> String {
        if self.status.starts_with("Running") {
            return self.footer_info_line();
        }
        if !self.selection_items.is_empty() {
            return "Enter select · Esc cancel".to_string();
        }
        let mut parts = Vec::new();
        if !self.follow_output {
            parts.push("PgUp/PgDn scroll".to_string());
            parts.push("End latest".to_string());
        }
        let footer_info = self.footer_info_line();
        if !footer_info.is_empty() {
            parts.push(footer_info);
        }
        parts.join(" · ")
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

    fn push_user_input(&mut self, text: &str) {
        self.committed_blocks.push(RenderBlock {
            lines: message_lines(USER_PREFIX, text, LineKind::User),
            kind: BlockKind::Conversation,
        });
        self.live_assistant = None;
        self.live_tool = None;
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
        self.scroll_top = 0;
        self.follow_output = true;
        self.unseen_output_lines = 0;
        self.rendered_lines.clear();
        self.rendered_log_height = 0;
        self.frozen_lines = None;
        self.status_since = None;
        for message in messages {
            self.push_message(message);
        }
    }

    fn apply_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::AgentStart => {}
            AgentEvent::TurnStart => {}
            AgentEvent::MessageStart { role } => match role {
                agent_model::LlmRole::Assistant => {
                    self.live_assistant = Some(RenderBlock {
                        lines: vec![StyledLine::new("", LineKind::Plain)],
                        kind: BlockKind::Conversation,
                    });
                }
                agent_model::LlmRole::User | agent_model::LlmRole::System => {}
                agent_model::LlmRole::Tool => {
                    self.live_tool = Some(RenderBlock {
                        lines: vec![StyledLine::new("· tool", LineKind::Tool)],
                        kind: BlockKind::Tool,
                    });
                }
            },
            AgentEvent::TextDelta(delta) => {
                if let Some(block) = self.live_assistant.as_mut() {
                    append_text_to_block(block, "", &delta);
                } else {
                    self.live_assistant = Some(RenderBlock {
                        lines: vec![StyledLine::new(delta, LineKind::Plain)],
                        kind: BlockKind::Conversation,
                    });
                }
            }
            AgentEvent::ToolCallStart { id, name } => {
                self.live_tool = Some(RenderBlock {
                    lines: vec![StyledLine::new(format!("· tool {name} ({id})"), LineKind::Tool)],
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
                    block.lines.push(StyledLine::new(format!("  done {id}"), LineKind::Tool));
                }
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
            AgentEvent::Usage(usage) => {
                self.latest_usage = Some(usage);
            }
            AgentEvent::TurnEnd { .. } => {}
            AgentEvent::AgentEnd => {
                self.live_assistant = None;
                self.live_tool = None;
            }
        }
    }

    fn active_log_lines(&self) -> &[StyledLine] {
        self.frozen_lines
            .as_ref()
            .map(|lines| lines.as_slice())
            .unwrap_or(self.rendered_lines.as_slice())
    }
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

fn apply_style(stdout: &mut Stdout, kind: LineKind) -> anyhow::Result<()> {
    match kind {
        LineKind::Plain | LineKind::Input => queue!(stdout, ResetColor)?,
        LineKind::User => queue!(
            stdout,
            SetForegroundColor(Color::White),
            SetBackgroundColor(Color::Rgb {
                r: 54,
                g: 58,
                b: 64,
            })
        )?,
        LineKind::System => queue!(stdout, SetForegroundColor(Color::DarkGrey), SetBackgroundColor(Color::Reset))?,
        LineKind::Tool => queue!(stdout, SetForegroundColor(Color::Cyan), SetBackgroundColor(Color::Reset))?,
        LineKind::Selection => queue!(
            stdout,
            SetForegroundColor(Color::White),
            SetBackgroundColor(Color::Rgb {
                r: 78,
                g: 82,
                b: 88,
            })
        )?,
        LineKind::Divider => queue!(stdout, SetForegroundColor(Color::DarkGrey), SetBackgroundColor(Color::Reset))?,
        LineKind::Status => queue!(stdout, SetForegroundColor(Color::Grey), SetBackgroundColor(Color::Reset))?,
        LineKind::Hint => queue!(stdout, SetForegroundColor(Color::DarkGrey), SetBackgroundColor(Color::Reset))?,
    }
    Ok(())
}

fn overlay_selection(
    frame: &mut [StyledLine],
    width: usize,
    height: usize,
    title: &str,
    items: &[String],
    selection_index: usize,
) {
    if width < 24 || height < 10 {
        return;
    }

    let content_width = items
        .iter()
        .map(|item| display_width(item))
        .max()
        .unwrap_or(0)
        .max(display_width(title))
        .min(width.saturating_sub(8));
    let preferred_width = width.saturating_sub(6);
    let box_width = (content_width + 6)
        .max(preferred_width.min(width.saturating_sub(4)))
        .clamp(36, width.saturating_sub(4));
    let available_height = height.saturating_sub(4);
    let content_capacity = available_height.saturating_sub(4).max(4);
    let visible_items = items.len().min(content_capacity);
    let min_box_height = (height.saturating_mul(2) / 3).clamp(8, available_height);
    let box_height = (visible_items + 4).max(min_box_height).min(available_height);
    let left = (width.saturating_sub(box_width)) / 2;
    let top = ((height.saturating_sub(box_height)).saturating_sub(1)) / 2;

    let max_scroll = items.len().saturating_sub(visible_items);
    let start = selection_index
        .saturating_sub(visible_items / 2)
        .min(max_scroll);
    let end = (start + visible_items).min(items.len());
    let horizontal = "─".repeat(box_width.saturating_sub(2));

    put_overlay_line(
        frame,
        top,
        left,
        width,
        &format!("┌{horizontal}┐"),
        LineKind::Divider,
    );
    put_overlay_line(
        frame,
        top + 1,
        left,
        width,
        &format!("│ {} │", pad_to_display_width(&clip_to_width(title, box_width.saturating_sub(4)), box_width.saturating_sub(4))),
        LineKind::Status,
    );
    put_overlay_line(
        frame,
        top + 2,
        left,
        width,
        &format!("├{horizontal}┤"),
        LineKind::Divider,
    );

    for (row_offset, item_index) in (start..end).enumerate() {
        let marker = if item_index == selection_index { "›" } else { " " };
        let item = &items[item_index];
        put_overlay_line(
            frame,
            top + 3 + row_offset,
            left,
            width,
            &format!(
                "│ {} {} │",
                marker,
                pad_to_display_width(
                    &clip_to_width(item, box_width.saturating_sub(6)),
                    box_width.saturating_sub(6)
                )
            ),
            if item_index == selection_index {
                LineKind::Selection
            } else {
                LineKind::Hint
            },
        );
    }

    for row in (top + 3 + (end - start))..(top + box_height - 1) {
        put_overlay_line(
            frame,
            row,
            left,
            width,
            &format!("│ {} │", " ".repeat(box_width.saturating_sub(4))),
            LineKind::Hint,
        );
    }

    put_overlay_line(
        frame,
        top + box_height - 1,
        left,
        width,
        &format!("└{horizontal}┘"),
        LineKind::Divider,
    );
}

fn put_overlay_line(
    frame: &mut [StyledLine],
    row: usize,
    left: usize,
    width: usize,
    text: &str,
    kind: LineKind,
) {
    if row >= frame.len() {
        return;
    }
    let prefix = " ".repeat(left);
    frame[row] = StyledLine::new(clip_to_width(&format!("{prefix}{text}"), width), kind);
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

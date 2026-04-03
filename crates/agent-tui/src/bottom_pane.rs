use crate::INPUT_PREFIX_WIDTH;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

pub(crate) fn render_input_lines(input: &str, width: usize) -> Vec<String> {
    let content_width = width.saturating_sub(INPUT_PREFIX_WIDTH).max(1);
    let mut lines = Vec::new();

    for raw_line in input.split('\n') {
        if raw_line.is_empty() {
            lines.push(String::new());
            continue;
        }

        lines.extend(wrap_line_by_display_width(raw_line, content_width));
    }

    if lines.is_empty() {
        lines.push(String::new());
    }

    lines
}

pub(crate) fn cursor_position_for_input(
    input: &str,
    cursor: usize,
    width: usize,
    input_start_row: u16,
) -> (u16, usize) {
    let content_width = width.saturating_sub(INPUT_PREFIX_WIDTH).max(1);
    let chars: Vec<char> = input.chars().collect();
    let cursor = cursor.min(chars.len());
    let (wrapped_row, wrapped_col) = wrapped_row_col_for_cursor(&chars, cursor, content_width);
    let row = input_start_row + wrapped_row as u16;
    let col = INPUT_PREFIX_WIDTH + wrapped_col;
    (row, col.min(width.saturating_sub(1)))
}

pub(crate) fn wrap_line_by_display_width(line: &str, width: usize) -> Vec<String> {
    let max_width = width.max(1);
    let mut wrapped = Vec::new();
    let mut current = String::new();
    let mut used = 0;

    for ch in line.chars() {
        let ch_width = display_width_char(ch).max(1);
        if !current.is_empty() && used + ch_width > max_width {
            wrapped.push(current);
            current = String::new();
            used = 0;
        }
        current.push(ch);
        used += ch_width;
        if used >= max_width {
            wrapped.push(current);
            current = String::new();
            used = 0;
        }
    }

    if current.is_empty() {
        if wrapped.is_empty() {
            wrapped.push(String::new());
        }
    } else {
        wrapped.push(current);
    }

    wrapped
}

fn wrapped_row_col_for_cursor(chars: &[char], cursor: usize, content_width: usize) -> (usize, usize) {
    let width = content_width.max(1);
    let mut row = 0;
    let mut col = 0;

    for ch in chars.iter().take(cursor) {
        if *ch == '\n' {
            row += 1;
            col = 0;
            continue;
        }

        let ch_width = display_width_char(*ch).max(1);
        if col > 0 && col + ch_width > width {
            row += 1;
            col = 0;
        }
        col += ch_width;
        if col >= width {
            row += col / width;
            col %= width;
        }
    }

    (row, col)
}

fn display_width_char(ch: char) -> usize {
    UnicodeWidthChar::width(ch).unwrap_or(0)
}

#[allow(dead_code)]
fn display_width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

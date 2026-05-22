# TUI Smoke Checklist

This checklist defines the current ratatui regression baseline.

## Automated Baseline

- `cargo test -p agent-tui`
- Fake stream coverage:
  - committed transcript is flushed into scrollback with `Viewport::Inline`
  - live assistant output remains above the prompt
  - prompt and footer stay in the bottom pane
  - multiline input does not push all live output away
  - queued prompt preview appears in the bottom pane while a run is active
  - wide overlay pickers show the selected item and preview column
  - provider mismatch text is split into readable overlay header lines

## Manual Provider Smoke

Run from the workspace root with a real provider configuration.

1. Plain prompt:
   - Assistant tokens stream in the live tail.
   - Final answer moves into committed transcript after completion.
   - Prompt remains directly above the footer without idle blank gaps.
2. Tool call:
   - In-flight tool call appears as a compact technical log.
   - Long tool output leaves a compact tail in transcript instead of flooding it.
3. Abort:
   - Press `Esc` during a run.
   - Partial run is discarded and a system note explains the abort.
4. Queued prompt:
   - Type a second prompt while the first run is active and press Enter.
   - Bottom pane shows `Queued next: ...`.
   - Queued prompt is submitted after the active run completes.
5. `/sessions`:
   - Current selection is highlighted.
   - Provider/model mismatch markers remain readable.
   - Wide terminals show a small preview column.
6. `/tree`:
   - Current checkpoint is highlighted.
   - Resize the terminal and confirm the selected row remains visible.
7. `/compact`:
   - Compaction result appears as a system note.
   - Transcript replay after reset remains readable.
8. Resize:
   - During streaming, resize narrower and wider.
   - Live tail, input, and footer stay in order without overlap.

## Surface Expectations

- Committed transcript: finalized user, assistant, tool, and system blocks only.
- Live tail: active assistant text and active tool log only.
- Bottom pane: queued prompt, editable input, status, footer, usage, and time.
- Overlay: modal selection UI; it may redraw the full inline viewport.


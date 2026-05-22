# TUI Model Redesign

This note defines the next TUI model for `miniature-agent`.

The current TUI is usable, but its rendering model is still too close to a
full-frame terminal app:

- transcript lines are recomputed and repainted as one frame
- the prompt/footer and transcript are not cleanly separated
- overlays are drawn on top of the same frame model
- terminal scrollback is not treated as a first-class output surface

That is the root cause of the remaining product gap. The next iteration should
change the model, not just tune the current drawing code.

## Goal

Make the chat surface behave like a terminal-native conversation document:

- committed transcript stays readable in terminal scrollback
- in-flight output is visible immediately without corrupting prior content
- the prompt remains stable and interactive while output streams
- overlays do not force a different mental model for the rest of the UI

This is closer to `pi`'s terminal-document direction and to the way Codex
separates committed transcript from the active in-flight cell.

## Non-goals

- No pane-heavy dashboard layout
- No general widget framework migration
- No attempt to make terminal scrollback, overlays, and live repaint all perfect
  in one step
- No new feature surface like autocomplete or `@` mentions until the base model
  is stable

## Current Problems

### 1. Transcript and footer share one frame model

The current renderer computes wrapped transcript lines and footer lines into one
`Vec<StyledLine>` and diffs that frame.

That works for correctness, but not for terminal-native history. When old lines
are redrawn as part of the same frame, scrollback cannot behave like a normal
terminal document.

### 2. In-flight output is not a separate rendering concept

The current state has block-level notions like `active_assistant` and
`active_tool`, but the renderer still treats the log as one repaint target.

Codex explicitly separates:

- committed transcript cells
- one active live cell
- an overlay-specific cached live tail

We need the same kind of boundary, even with a much smaller implementation.

### 3. Overlay and normal conversation use the same mental model

The overlay works, but it is still just a different drawing mode on top of the
same frame. That is fine for now, but it means the main transcript cannot
evolve toward append-friendly rendering independently.

### 4. Scrollback is not a first-class surface

We treated scrollback as a byproduct of the frame renderer. It needs to become a
deliberate output target.

## Target Model

The TUI should be modeled as four independent surfaces.

### 1. Committed transcript

This is append-only from the TUI's point of view.

- Contains finalized user / assistant / tool / system blocks
- Once flushed, it is not repainted in place during normal chat flow
- Intended to live in terminal scrollback

This is the primary reading surface.

### 2. Live tail

This is the only mutable chat output area.

- Contains the current in-flight assistant block and any in-flight tool block
- Lives above the prompt/footer
- Repaints in place while streaming
- Is converted into committed transcript when the turn finalizes

This is the key missing abstraction in the current codebase.

### 3. Bottom pane

This is the stable input and status surface.

- Prompt input
- running / aborted / failed status
- footer info like path, model, usage, time
- minimal hints

This surface is allowed to repaint freely.

### 4. Overlay

This is temporary and modal.

- sessions picker
- tree picker
- later: pager / transcript preview / file mention chooser

Overlay can still use a full-frame redraw model if needed. It should not force
the normal chat surface to do the same.

## State Model

The TUI state should be split accordingly.

### Transcript state

- `committed_blocks: Vec<RenderBlock>`
- `committed_height_cache` or equivalent optional wrapping cache
- no concept of "editing" these blocks during normal rendering

### Live state

- `active_assistant: Option<LiveBlock>`
- `active_tool: Option<LiveBlock>`
- maybe one `live_tail: Vec<RenderBlock>` if that is simpler

Only this state mutates rapidly during streaming.

### Bottom-pane state

- input buffer
- cursor
- running status
- footer info
- queued prompt status

### Overlay state

- title
- items
- selection index
- maybe preview text later

## Rendering Model

## Phase 1: Stable split without full scrollback ambitions

First, split rendering responsibilities without changing too much behavior.

- render transcript lines from committed blocks only
- render live tail separately
- render bottom pane separately
- keep overlay as a full-frame path

This makes the data model correct before changing the terminal write strategy.

## Phase 2: Append-first transcript flushing

After the split is stable:

- newly committed transcript blocks are written to terminal output once
- only live tail and bottom pane are repainted
- committed transcript is no longer part of the normal frame diff loop

This is the first phase where terminal scrollback should materially improve.

Important constraint:

- do not attempt to retroactively redraw committed transcript when width changes
- width changes should affect future rendering, live tail, overlays, and prompt
- already-flushed transcript may keep old wrapping, which is acceptable

That tradeoff is worth it if scrollback becomes dependable.

## Phase 3: Overlay/pager coexistence

Once append-first transcript is stable:

- overlays can temporarily take over the bottom of the screen
- transcript itself remains committed in scrollback
- a dedicated pager overlay can later be added for transcript browsing

This is a better fit than trying to make the main chat surface itself both a
full-screen viewport and a terminal scrollback document at the same time.

## Why not ratatui right now?

The remaining problems are not mainly widget or layout problems.

They are:

- transcript commitment semantics
- mutable live tail semantics
- footer ownership
- scrollback ownership

`ratatui` could help later if the UI becomes pane-heavy, but it does not remove
the need for these state boundaries. Right now, a cleaner model on top of
`crossterm` is the more direct path.

## References and Takeaways

### Codex

Useful pattern:

- committed transcript cells
- one active in-flight cell
- overlay uses committed transcript plus a cached live tail

We should copy the idea, not the full complexity.

### pi_agent_rust

Useful pattern:

- conversation rendering is kept separate from other interactive concerns
- terminal view logic is treated as its own layer

We should keep that kind of module boundary.

## First Implementation Slice

Do not jump straight to append-first flushing again. The previous attempt showed
that doing the terminal-write transition before stabilizing the state split is
too risky.

First slice:

1. Introduce explicit `committed_blocks` and `live_blocks`
2. Keep full-frame redraw for now
3. Render transcript from committed only
4. Render live tail as a separate section above the prompt
5. Keep overlays on the current full-frame path

This is the smallest change that moves the model in the right direction.

If that slice feels stable in use, only then move to append-first transcript
flushing.

## Decision

The next TUI work should optimize for this sequence:

1. state split
2. render split
3. append-first transcript flush
4. richer bottom pane and overlays
5. code highlighting and diff

This keeps the hardest problem isolated and avoids mixing scrollback work with
input-feature work.

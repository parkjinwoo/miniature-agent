# Modernization Plan

## Product Direction

`miniature-agent` should become a small, durable, local-first coding-agent runtime.
Its advantage is not the number of integrations it has. Its advantage is that a user can
understand what happened, resume work safely, and trust the saved history.

The project should optimize for:

- inspectability over hidden orchestration
- durable sessions over disposable chat transcripts
- explicit trust boundaries over unrestricted automation
- provider portability over provider-specific product features
- deterministic recovery and tests over feature count

This is a better fit for the project than trying to become a general agent platform,
plugin marketplace, IDE, or multi-agent coordinator.

## Current Baseline

The codebase already has valuable foundations:

- provider-neutral streaming and tool events
- native OpenAI Responses and Anthropic Messages adapters
- an OpenAI-compatible Chat Completions adapter
- append-only, branchable JSONL sessions
- summary compaction
- interactive and non-interactive entry points
- a terminal UI with a committed transcript and live tail

The first modernization pass also established these reliability guards:

- OpenAI function-call history is preserved across tool round trips
- Responses requests opt out of server-side storage
- a per-prompt model-turn limit prevents infinite tool loops
- malformed tool-continuation responses stop as errors
- forks copy only the selected session branch
- a torn final JSONL record can be recovered after interruption
- session versions, entry ordering, IDs, and parents are validated
- file reads and tool output are bounded before returning to the model
- non-interactive runs are read-only unless `--full-access` is explicitly passed
- shell commands have bounded capture, timeouts, and process-group cleanup
- text edits reject empty or ambiguous targets
- CLI arguments fail clearly instead of silently selecting a fallback
- non-interactive runs fork a resumed transcript before crossing a provider/model boundary
- formatting, Clippy, and workspace tests run in CI

## Roadmap

### P0: Trust boundary and execution safety

The current `write`, `edit`, and `bash` tools execute automatically. That is acceptable for
an experimental agent, but not a strong default for a tool people trust with real repositories.

Implement next:

1. Add an interactive `ask` policy alongside the existing non-interactive `read-only` and
   `full-access` policies.
2. Make `ask` the interactive default and preview file diffs and shell commands before approval.
3. Record policy, approvals, denials, exit codes, duration, and truncation in session events.
4. Add a structured patch tool; keep exact-string edit as a small fallback.

Exit criteria:

- no mutating action happens without matching policy or approval
- a hung or noisy command cannot hang the agent or exhaust memory
- every side effect has a durable audit record

### P1: Session durability and migrations

Append-only JSONL is the product's strongest differentiator and should be treated as a data format,
not an implementation detail.

Implement next:

1. Persist the selected leaf independently so navigation survives restart without a new turn.
2. Add a `doctor` command that reports and repairs recoverable session tails.
3. Introduce explicit migrations before changing the session schema.
4. Add integrity metadata for records and atomic replacement for rewritten files.
5. Separate replay-critical records from high-volume UI telemetry.
6. Test crashes at every write boundary and test branch/compact/fork combinations property-wise.

Exit criteria:

- interrupted writes never hide a usable session
- every supported historical version has a tested migration path
- branching, compaction, and forking preserve exactly one intended lineage

### P1: Provider conformance

Provider APIs evolve independently. The adapters need conformance tests rather than a growing set
of one-off compatibility flags.

Implement next:

1. Build fixture suites from documented event shapes for text, usage, refusal, cancellation,
   multiple tool calls, partial JSON, and truncated streams.
2. Add capability negotiation for reasoning, images, strict schemas, and parallel tool calls.
3. Validate base URLs and model configuration with a `doctor --provider` command.
4. Keep model names configurable; avoid frequent releases only to chase default model aliases.
5. Add optional JSON event output for scripts and integration tests.

Exit criteria:

- every adapter passes the same provider-neutral behavior suite
- unsupported capabilities fail before a request is sent
- provider failures retain status and response details without leaking credentials

### P2: Context quality

Message-count compaction is simple but does not correspond to model context limits.

Implement next:

1. Track token estimates and provider-reported usage per branch.
2. Compact by a configurable context budget rather than a fixed message count.
3. Produce structured summaries with decisions, changed files, commands, failures, and open work.
4. Preserve important tool evidence separately from conversational prose.
5. Add summary evaluation fixtures so compaction quality can improve without silent regressions.

Exit criteria:

- context size is predictable before each request
- a resumed task retains decisions and outstanding work, not merely a prose synopsis

### P2: Operator experience

Implement after the trust and durability work:

1. Replace the hand-written CLI parser with a declarative parser when the command surface grows.
2. Add `doctor`, machine-readable `--json`, and explicit session selection/resume flags.
3. Search sessions by workspace, model, time, and transcript text.
4. Improve diff and code rendering while keeping the terminal-document model.
5. Publish reproducible release binaries with checksums and a minimal install path.

## Deliberate Non-goals

Until the priorities above are complete, do not add:

- multi-agent orchestration
- a general plugin or connector ecosystem
- background autonomous work
- browser/computer control
- a hosted account or synchronization service
- a pane-heavy IDE replacement

Those features enlarge the attack surface and operating complexity without strengthening the
project's distinctive session-first core.

## Release Strategy

- `0.2`: reliability baseline from this pass, CI, and documented limits
- `0.3`: approval policy, command containment, and structured patching
- `0.4`: session doctor, migrations, and crash-consistency suite
- `0.5`: provider conformance and machine-readable event mode
- `1.0`: stable session format, safe defaults, documented compatibility, reproducible releases

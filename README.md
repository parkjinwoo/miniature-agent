# miniature-agent

A miniature agent for distilling context.

`miniature-agent` is a small coding agent.
Its purpose is not just to keep a flat transcript, but to give you an environment where work can be resumed with useful memory, revisited, forked, compacted, and learned from over time.

## What It Supports

- OpenAI `Responses`
- Anthropic `Messages`
- a generic `compatible` provider for OpenAI-compatible `Chat Completions` backends
- append-only JSONL sessions with resume, fork, checkpoint selection, and summary-based compaction
- built-in `read`, `write`, `edit`, and `bash` tools
- a `crossterm` TUI and a non-interactive `--prompt` mode
- recovery from interrupted trailing session writes and branch-correct session forks
- bounded model turns, file reads, and returned tool output

## Install

This project is not packaged for Homebrew or other system package managers yet.
The expected setup is:

1. build the binary from source
2. place or symlink it somewhere on your `PATH`

Example:

```bash
cargo build --release -p agent-cli
mkdir -p ~/.local/bin
ln -sf "$PWD/target/release/agent-cli" ~/.local/bin/miniature-agent
```

Then run:

```bash
miniature-agent
```

Run `miniature-agent --help` for the complete command list. Invalid or incomplete flags fail with
an actionable error instead of silently falling back to another provider or mode.

## Config And Session Paths

- config file: `~/.config/miniature-agent/config.toml`
- state dir: `~/.local/state/miniature-agent/`
- sessions: `~/.local/state/miniature-agent/sessions/`

Useful commands:

- `miniature-agent --print-paths`
- `miniature-agent --write-default-config`
- `/paths` inside the TUI

For provider base URLs, precedence is:
- `config.toml` `base_url`
- provider-specific `*_BASE_URL` environment variable
- built-in default

## First Run

1. Write the default config once:

```bash
miniature-agent --write-default-config
```

2. Inspect where config and sessions will live:

```bash
miniature-agent --print-paths
```

3. Add a provider key if needed:

- `MINIATURE_AGENT_OPENAI_API_KEY`
- `MINIATURE_AGENT_ANTHROPIC_API_KEY`
- `MINIATURE_AGENT_COMPATIBLE_API_KEY` if your compatible backend requires one

Example:

```bash
export MINIATURE_AGENT_OPENAI_API_KEY="your-openai-api-key"
export MINIATURE_AGENT_ANTHROPIC_API_KEY="your-anthropic-api-key"
```

4. Start the TUI:

```bash
miniature-agent
```

5. Or run one prompt without entering the TUI:

```bash
miniature-agent --provider openai --prompt "read src/main.rs"
```

Non-interactive prompt mode is read-only by default. To deliberately expose `write`, `edit`, and
`bash`, add `--full-access`:

```bash
miniature-agent --provider openai --prompt "fix the failing tests" --full-access
```

## Minimal Config

The default config is standard TOML at `~/.config/miniature-agent/config.toml`.

Example:

```toml
[providers.openai]
base_url = "https://your-openai-compatible-endpoint.example/v1/responses"
default_model = "your-openai-model"

[providers.anthropic]
base_url = "https://your-anthropic-compatible-endpoint.example/v1/messages"
default_model = "your-anthropic-model"

[providers.compatible]
base_url = "https://your-openai-compatible-endpoint.example/v1/chat/completions"
default_model = "your-model-name"
requires_api_key = true
```

For a local server, use the same `providers.compatible` block with a local `base_url` and set `requires_api_key = false`.

In practice, most users only need:
- a provider API key in the environment
- an optional model override in `config.toml`

The intended split is:
- environment variables hold secrets
- `config.toml` holds provider, model, and endpoint choices

If you prefer environment overrides for endpoints, these are also supported:
- `MINIATURE_AGENT_OPENAI_BASE_URL`
- `MINIATURE_AGENT_ANTHROPIC_BASE_URL`
- `MINIATURE_AGENT_COMPATIBLE_BASE_URL`

If you set `session_dir` manually, use an absolute path. The default XDG session location is usually the simplest choice.

## Common Commands

CLI:

- `miniature-agent --list-sessions`
- `miniature-agent --provider anthropic`
- `miniature-agent --provider compatible --prompt "hello"`

Inside the TUI:

- `/help`
- `/sessions`
- `/tree`
- `/fork`
- `/provider`
- `/paths`
- `/compact`

## Safety Model

The file tools reject paths that escape the starting workspace, including escapes through existing
symbolic links. Reads larger than 1 MiB are rejected, captured shell output is memory-bounded, and
a prompt is stopped after 16 model turns to prevent an unbounded tool loop. Shell commands default
to a 30-second timeout, can request at most 120 seconds, and have their remaining process group
terminated when the command exits or times out.

Non-interactive `--prompt` runs expose only `read` unless `--full-access` is passed. The interactive
TUI still executes `write`, `edit`, and `bash` calls without a per-action approval step, so use it
only in a workspace and account you are comfortable allowing the model to modify. An interactive
`ask` policy with command and diff previews is the highest-priority follow-up work.

## Verification

- `cargo check`
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- `miniature-agent --print-paths`
- `miniature-agent --provider <name> --prompt "hello"`

The minimum supported Rust version is 1.88. CI runs the locked dependency graph on Ubuntu 24.04
x86_64, Intel macOS, and Apple Silicon macOS. The command runner intentionally depends on POSIX
process-group behavior, so Windows is not currently supported. See
[`docs/platform-support.md`](docs/platform-support.md) for the support policy and platform-specific
details.

Before tagging a release, follow [`docs/v1-release-checklist.md`](docs/v1-release-checklist.md).
For the current project direction and non-goals, see [`docs/project-scope.md`](docs/project-scope.md).
For the modernization priorities and release sequence, see
[`docs/modernization-plan.md`](docs/modernization-plan.md).

## Workspace

- `crates/agent-model`: provider-neutral model types plus OpenAI `Responses`, Anthropic `Messages`, and generic compatible `Chat Completions`
- `crates/agent-core`: event-driven agent loop
- `crates/agent-session`: append-only JSONL session storage
- `crates/agent-tools`: built-in `read` / `write` / `edit` / `bash`
- `crates/agent-tui`: terminal UI primitives
- `crates/agent-cli`: XDG-aware CLI, session navigation, provider selection, and app orchestration

## Credits

This project was developed with substantial LLM assistance and draws clear conceptual inspiration from `pi` by Mario Zechner.

In particular, the session-first workflow, append-only history, and terminal-oriented coding-agent experience were shaped by studying `pi` and related writing.

- `pi`: <https://github.com/badlogic/pi-mono>
- Mario Zechner, “What I learned building an opinionated and minimal coding agent”
- Armin Ronacher, “Pi: The Minimal Agent Within OpenClaw”

## License

The project source is licensed under MIT. See [`LICENSE`](LICENSE).

Dependency licenses were checked with `cargo-license`.
The current dependency tree is mostly MIT / Apache-2.0 family, with a smaller number of other permissive licenses such as ISC, BSD-3-Clause, Unicode-3.0, Zlib, and CDLA-Permissive-2.0.

I do not currently see an immediate release blocker from the dependency licenses, but anyone shipping this more widely should still review the generated dependency report for their own distribution needs.

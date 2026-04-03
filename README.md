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

## Verification

- `cargo check`
- `cargo test --workspace`
- `miniature-agent --print-paths`
- `miniature-agent --provider <name> --prompt "hello"`

Before tagging a release, follow [`docs/v1-release-checklist.md`](docs/v1-release-checklist.md).
For the current project direction and non-goals, see [`docs/project-scope.md`](docs/project-scope.md).

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

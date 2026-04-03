# v1 Release Checklist

## Automated Gate

- Run `cargo check`
- Run `cargo test --workspace`
- Confirm no unexpected local diff in generated session files or temp fixtures
- Run `cargo run -p agent-cli -- --print-paths`
- Confirm config and session locations match the expected XDG paths

## Basic Manual Smoke

Run each item from a clean workspace copy if possible.

0. Run `cargo run -p agent-cli -- --write-default-config` if no config exists yet.
0. Run `cargo run -p agent-cli -- --print-paths` and confirm config/state locations are readable and expected.
1. Start the CLI with a real provider key.
2. Submit a plain text prompt and confirm streamed assistant output appears in the TUI.
3. Ask for a file read and confirm the `read` tool result is shown and persisted.
4. Ask for a file write or edit inside the workspace and confirm the file changed as expected.
5. Abort an in-flight response with `Esc` and confirm the session does not append a partial run.
6. Re-open the same session and confirm transcript replay matches the saved session file.

## Non-Interactive Smoke

- Run `cargo run -p agent-cli -- --provider <name> --prompt "hello"`
- Confirm the command either prints a streamed turn result or fails with a provider/network error instead of a TUI/raw-mode error

## Session Smoke

1. Create a new session and send at least two turns.
2. Run `/tree` and confirm checkpoints appear.
3. Run `/compact` and confirm a summary entry is inserted while recent turns remain visible.
4. Run `/sessions` and confirm the current session is listed with provider/model info.
5. Run `/paths` and confirm the shown config/state/session paths match the expected XDG locations.
6. Run `/fork` and confirm a new session file is created with the current provider snapshot.

## Provider Smoke

### OpenAI

- Set `MINIATURE_AGENT_OPENAI_API_KEY`
- Run default provider flow
- Confirm text streaming and tool calling both work

### Anthropic

- Set `MINIATURE_AGENT_ANTHROPIC_API_KEY`
- Run `cargo run -p agent-cli -- --provider anthropic`
- Confirm text streaming and tool calling both work

### OpenAI-compatible Chat Completions

- Use `--provider compatible`
- Set `MINIATURE_AGENT_COMPATIBLE_BASE_URL` or `config.toml` `base_url` to your compatible endpoint
- If the backend requires auth, set `MINIATURE_AGENT_COMPATIBLE_API_KEY`
- Local servers may leave `MINIATURE_AGENT_COMPATIBLE_API_KEY` unset
- Confirm provider/model mismatch markers in `/sessions` look correct
- Confirm at least one tool call round-trip succeeds

## Provider Mismatch Smoke

1. Open a session created with a different provider or model.
2. Confirm the picker marks it as mismatch.
3. Confirm the mismatch prompt recommends `fork`.
4. Choose the recommended fork path and confirm the new session header stores the current provider snapshot.

## Known Risks Before v1 Tag

- The TUI is usable but still not a fully polished scrollback-preserving renderer.
- Abort handling is stable at the state level, but real provider latency and cancellation timing should be manually checked.
- OpenAI-compatible providers vary in stream shape; fixture coverage is good, but real smoke runs still matter.
- The built-in `edit` tool is intentionally minimal and may need a richer patch workflow after v1.

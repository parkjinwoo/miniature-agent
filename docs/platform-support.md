# Platform support

## Supported environments

The supported Rust toolchain is stable Rust 1.88 or newer. The repository keeps a locked dependency
graph and verifies it in CI on these native environments:

| Environment | Rust target | CI runner |
| --- | --- | --- |
| Linux x86_64 | `x86_64-unknown-linux-gnu` | Ubuntu 24.04 |
| Intel macOS | `x86_64-apple-darwin` | macOS 15 Intel |
| Apple Silicon macOS | `aarch64-apple-darwin` | macOS 15 ARM64 |

Intel and Apple Silicon use the same Rust source and configuration. No architecture-specific files,
prebuilt native libraries, or hard-coded Homebrew paths are required.

## Operating-system assumptions

- The `bash` tool launches `$SHELL` when it is set and otherwise launches `/bin/sh`.
- Shell commands run in a separate POSIX process group so a timeout can terminate descendants as
  well as the shell itself. This behavior is covered on Linux and both macOS architectures by the
  native CI test suite.
- Configuration and session data follow `XDG_CONFIG_HOME` and `XDG_STATE_HOME`. If they are unset,
  the fallback locations are `~/.config/miniature-agent` and
  `~/.local/state/miniature-agent`, consistently on Linux and macOS.
- Network providers use Rustls instead of a system OpenSSL installation. This removes a common
  source of architecture- and distribution-specific build failures.

Windows is not supported at present because process containment depends on Unix process groups and
signals. Adding Windows support requires a Job Object-based command runner rather than compiling out
the current containment behavior.

## Local verification

Run the complete native verification suite with:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --locked
```

On an Intel Mac with Xcode installed, Apple Silicon compilation can also be checked without running
the binary:

```bash
rustup target add aarch64-apple-darwin
cargo check --workspace --all-targets --locked --target aarch64-apple-darwin
```

If `rustup` reports that the target is installed but Cargo cannot find the `core` crate, check
`command -v cargo`, `command -v rustc`, and `rustc --print sysroot`. A separately installed
Homebrew Rust can take precedence over the rustup-managed toolchain; put the rustup shims first on
`PATH` before retrying.

Native CI remains the release gate because it exercises architecture-specific linking and runtime
behavior that a compile-only cross-check cannot prove.

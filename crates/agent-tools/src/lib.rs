use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::Component;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Context;
use schemars::Schema;
use serde::Deserialize;

use agent_model::ToolSpec;

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments_json: String,
}

#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    fn run(&self, call: &ToolCall) -> anyhow::Result<ToolOutput>;
}

pub struct ToolRegistry {
    tools: BTreeMap<String, Box<dyn Tool>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: BTreeMap::new(),
        }
    }

    pub fn register<T>(&mut self, tool: T)
    where
        T: Tool + 'static,
    {
        self.tools.insert(tool.spec().name.clone(), Box::new(tool));
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.values().map(|tool| tool.spec()).collect()
    }

    pub fn execute(&self, call: &ToolCall) -> anyhow::Result<ToolOutput> {
        let tool = self
            .tools
            .get(&call.name)
            .with_context(|| format!("unknown tool: {}", call.name))?;
        tool.run(call)
    }
}

const MAX_READ_BYTES: u64 = 1_048_576;
const MAX_TOOL_OUTPUT_BYTES: usize = 262_144;
const DEFAULT_COMMAND_TIMEOUT_SECONDS: u64 = 30;
const MAX_COMMAND_TIMEOUT_SECONDS: u64 = 120;

pub fn default_tool_registry(cwd: impl Into<PathBuf>) -> ToolRegistry {
    let cwd = cwd.into();
    let mut registry = ToolRegistry::new();
    registry.register(ReadTool { cwd: cwd.clone() });
    registry.register(WriteTool { cwd: cwd.clone() });
    registry.register(EditTool { cwd: cwd.clone() });
    registry.register(BashTool { cwd });
    registry
}

pub fn read_only_tool_registry(cwd: impl Into<PathBuf>) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(ReadTool { cwd: cwd.into() });
    registry
}

fn resolve_path(base: &Path, relative: &str) -> anyhow::Result<PathBuf> {
    let canonical_base = base
        .canonicalize()
        .or_else(|_| Ok::<PathBuf, anyhow::Error>(base.to_path_buf()))?;
    let joined = normalize_joined_path(&canonical_base, relative)?;
    let mut current = canonical_base.clone();

    for component in Path::new(relative).components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                anyhow::bail!("path escapes workspace: {}", relative);
            }
            Component::Normal(part) => {
                current.push(part);
                if current.exists() {
                    let canonical_current = current.canonicalize()?;
                    if !canonical_current.starts_with(&canonical_base) {
                        anyhow::bail!("path escapes workspace: {}", relative);
                    }
                    current = canonical_current;
                }
            }
        }
    }

    if !current.starts_with(&canonical_base) {
        anyhow::bail!("path escapes workspace: {}", relative);
    }

    Ok(joined)
}

fn normalize_joined_path(base: &Path, relative: &str) -> anyhow::Result<PathBuf> {
    let relative_path = Path::new(relative);
    if relative_path.is_absolute() {
        anyhow::bail!("path escapes workspace: {}", relative);
    }

    let mut normalized = base.to_path_buf();
    for component in relative_path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if normalized == base || !normalized.pop() {
                    anyhow::bail!("path escapes workspace: {}", relative);
                }
            }
            Component::Normal(_) => {
                normalized.push(component.as_os_str());
            }
            Component::RootDir | Component::Prefix(_) => {
                anyhow::bail!("path escapes workspace: {}", relative);
            }
        }
    }

    if !normalized.starts_with(base) {
        anyhow::bail!("path escapes workspace: {}", relative);
    }

    Ok(normalized)
}

fn build_schema<T>() -> Schema
where
    T: schemars::JsonSchema,
{
    schemars::schema_for!(T)
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ReadArgs {
    path: String,
}

pub struct ReadTool {
    cwd: PathBuf,
}

impl Tool for ReadTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read".to_string(),
            description: "Read a UTF-8 text file from the workspace".to_string(),
            input_schema: serde_json::to_value(build_schema::<ReadArgs>())
                .unwrap_or(serde_json::json!({})),
        }
    }

    fn run(&self, call: &ToolCall) -> anyhow::Result<ToolOutput> {
        let args: ReadArgs = serde_json::from_str(&call.arguments_json)?;
        let path = resolve_path(&self.cwd, &args.path)?;
        let metadata =
            fs::metadata(&path).with_context(|| format!("failed to inspect {}", path.display()))?;
        anyhow::ensure!(
            metadata.len() <= MAX_READ_BYTES,
            "refusing to read {} bytes from {}; limit is {} bytes",
            metadata.len(),
            path.display(),
            MAX_READ_BYTES
        );
        let content = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        Ok(ToolOutput {
            content,
            is_error: false,
        })
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct WriteArgs {
    path: String,
    content: String,
}

pub struct WriteTool {
    cwd: PathBuf,
}

impl Tool for WriteTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write".to_string(),
            description: "Write a UTF-8 text file in the workspace".to_string(),
            input_schema: serde_json::to_value(build_schema::<WriteArgs>())
                .unwrap_or(serde_json::json!({})),
        }
    }

    fn run(&self, call: &ToolCall) -> anyhow::Result<ToolOutput> {
        let args: WriteArgs = serde_json::from_str(&call.arguments_json)?;
        let path = resolve_path(&self.cwd, &args.path)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, args.content)
            .with_context(|| format!("failed to write {}", path.display()))?;
        Ok(ToolOutput {
            content: format!("Wrote {}", path.display()),
            is_error: false,
        })
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct EditArgs {
    path: String,
    old: String,
    new: String,
}

pub struct EditTool {
    cwd: PathBuf,
}

impl Tool for EditTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "edit".to_string(),
            description: "Replace one string with another inside a UTF-8 file".to_string(),
            input_schema: serde_json::to_value(build_schema::<EditArgs>())
                .unwrap_or(serde_json::json!({})),
        }
    }

    fn run(&self, call: &ToolCall) -> anyhow::Result<ToolOutput> {
        let args: EditArgs = serde_json::from_str(&call.arguments_json)?;
        let path = resolve_path(&self.cwd, &args.path)?;
        let content = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;

        anyhow::ensure!(!args.old.is_empty(), "edit target must not be empty");
        if !content.contains(&args.old) {
            anyhow::bail!("target text not found in {}", path.display());
        }
        let match_count = content.match_indices(&args.old).take(2).count();
        anyhow::ensure!(
            match_count == 1,
            "target text occurs more than once in {}; provide a unique target",
            path.display()
        );

        let updated = content.replacen(&args.old, &args.new, 1);
        fs::write(&path, updated).with_context(|| format!("failed to write {}", path.display()))?;

        Ok(ToolOutput {
            content: format!("Edited {}", path.display()),
            is_error: false,
        })
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct BashArgs {
    command: String,
    /// Command timeout in seconds. Defaults to 30 and is capped at 120.
    timeout_seconds: Option<u64>,
}

pub struct BashTool {
    cwd: PathBuf,
}

impl Tool for BashTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "bash".to_string(),
            description: "Run a shell command in the workspace with a bounded timeout and output"
                .to_string(),
            input_schema: serde_json::to_value(build_schema::<BashArgs>())
                .unwrap_or(serde_json::json!({})),
        }
    }

    fn run(&self, call: &ToolCall) -> anyhow::Result<ToolOutput> {
        let args: BashArgs = serde_json::from_str(&call.arguments_json)?;
        let shell = std::env::var("SHELL")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "/bin/sh".to_string());
        let timeout_seconds = args
            .timeout_seconds
            .unwrap_or(DEFAULT_COMMAND_TIMEOUT_SECONDS)
            .clamp(1, MAX_COMMAND_TIMEOUT_SECONDS);
        let mut command = Command::new(&shell);
        command
            .arg("-lc")
            .arg(&args.command)
            .current_dir(&self.cwd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let mut child = command
            .spawn()
            .with_context(|| format!("failed to run command with {}: {}", shell, args.command))?;
        let stdout_reader = bounded_reader(
            child
                .stdout
                .take()
                .context("failed to capture command stdout")?,
            MAX_TOOL_OUTPUT_BYTES,
        );
        let stderr_reader = bounded_reader(
            child
                .stderr
                .take()
                .context("failed to capture command stderr")?,
            MAX_TOOL_OUTPUT_BYTES,
        );

        let deadline = Instant::now() + Duration::from_secs(timeout_seconds);
        let (status, timed_out) = loop {
            if let Some(status) = child.try_wait().context("failed to poll shell command")? {
                kill_remaining_process_group(child.id());
                break (status, false);
            }
            if Instant::now() >= deadline {
                terminate_process_tree(&mut child);
                let status = child
                    .wait()
                    .context("failed to wait for timed out shell command")?;
                break (status, true);
            }
            thread::sleep(Duration::from_millis(20));
        };

        let stdout = join_reader(stdout_reader, "stdout")?;
        let stderr = join_reader(stderr_reader, "stderr")?;
        let mut content = String::from_utf8_lossy(&stdout.bytes).into_owned();
        if stdout.truncated {
            content.push_str("\n[stdout capture truncated]");
        }
        if !stderr.bytes.is_empty() {
            if !content.is_empty() {
                content.push('\n');
            }
            content.push_str(&String::from_utf8_lossy(&stderr.bytes));
        }
        if stderr.truncated {
            content.push_str("\n[stderr capture truncated]");
        }
        if content.is_empty() {
            content = "(no output)".to_string();
        }
        if timed_out {
            if !content.ends_with('\n') {
                content.push('\n');
            }
            content.push_str(&format!("[command timed out after {timeout_seconds}s]"));
        } else if !status.success() {
            if !content.ends_with('\n') {
                content.push('\n');
            }
            content.push_str(&format!("[command exited with status {status}]"));
        }
        truncate_utf8(&mut content, MAX_TOOL_OUTPUT_BYTES);

        Ok(ToolOutput {
            content,
            is_error: timed_out || !status.success(),
        })
    }
}

struct BoundedOutput {
    bytes: Vec<u8>,
    truncated: bool,
}

fn bounded_reader(
    mut reader: impl Read + Send + 'static,
    limit: usize,
) -> thread::JoinHandle<std::io::Result<BoundedOutput>> {
    thread::spawn(move || {
        let mut bytes = Vec::with_capacity(limit.min(8192));
        let mut truncated = false;
        let mut buffer = [0u8; 8192];
        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            let remaining = limit.saturating_sub(bytes.len());
            let keep = remaining.min(read);
            bytes.extend_from_slice(&buffer[..keep]);
            truncated |= keep < read;
        }
        Ok(BoundedOutput { bytes, truncated })
    })
}

fn join_reader(
    handle: thread::JoinHandle<std::io::Result<BoundedOutput>>,
    stream_name: &str,
) -> anyhow::Result<BoundedOutput> {
    handle
        .join()
        .map_err(|_| anyhow::anyhow!("{stream_name} capture thread panicked"))?
        .with_context(|| format!("failed to read command {stream_name}"))
}

#[cfg(unix)]
fn kill_remaining_process_group(pid: u32) {
    // SAFETY: `pid` came from a child that was placed in its own process group. A negative PID
    // targets only that group; errors are intentionally ignored when the group is already gone.
    unsafe {
        libc::kill(-(pid as i32), libc::SIGKILL);
    }
}

#[cfg(not(unix))]
fn kill_remaining_process_group(_: u32) {}

fn terminate_process_tree(child: &mut Child) {
    kill_remaining_process_group(child.id());
    let _ = child.kill();
}

fn truncate_utf8(value: &mut String, max_bytes: usize) {
    if value.len() <= max_bytes {
        return;
    }

    let suffix = "\n[output truncated]";
    let mut keep = max_bytes.saturating_sub(suffix.len());
    while keep > 0 && !value.is_char_boundary(keep) {
        keep -= 1;
    }
    value.truncate(keep);
    value.push_str(suffix);
}

#[cfg(test)]
mod tests {
    use super::*;
    fn temp_workspace() -> PathBuf {
        let unique = uuid::Uuid::new_v4();
        let path = std::env::temp_dir().join(format!("miniature-agent-tools-{unique}"));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn call(name: &str, arguments_json: &str) -> ToolCall {
        ToolCall {
            id: "call-1".to_string(),
            name: name.to_string(),
            arguments_json: arguments_json.to_string(),
        }
    }

    #[test]
    fn write_read_and_edit_roundtrip() {
        let cwd = temp_workspace();
        let registry = default_tool_registry(&cwd);

        let write = registry
            .execute(&call(
                "write",
                r#"{"path":"notes/todo.txt","content":"before"}"#,
            ))
            .unwrap();
        assert!(!write.is_error);

        let edited = registry
            .execute(&call(
                "edit",
                r#"{"path":"notes/todo.txt","old":"before","new":"after"}"#,
            ))
            .unwrap();
        assert!(!edited.is_error);

        let read = registry
            .execute(&call("read", r#"{"path":"notes/todo.txt"}"#))
            .unwrap();
        assert_eq!(read.content, "after");

        let _ = fs::remove_dir_all(cwd);
    }

    #[test]
    fn read_only_registry_exposes_no_mutating_tools() {
        let cwd = temp_workspace();
        let registry = read_only_tool_registry(&cwd);
        let specs = registry.specs();

        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "read");
        assert!(
            registry
                .execute(&call("bash", r#"{"command":"touch changed"}"#))
                .unwrap_err()
                .to_string()
                .contains("unknown tool")
        );
        assert!(!cwd.join("changed").exists());

        let _ = fs::remove_dir_all(cwd);
    }

    #[test]
    fn rejects_paths_outside_workspace() {
        let cwd = temp_workspace();
        let registry = default_tool_registry(&cwd);

        let error = registry
            .execute(&call(
                "write",
                r#"{"path":"../escape.txt","content":"bad"}"#,
            ))
            .unwrap_err()
            .to_string();
        assert!(error.contains("path escapes workspace"));

        let _ = fs::remove_dir_all(cwd);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_writes_through_symlink_that_escapes_workspace() {
        use std::os::unix::fs::symlink;

        let cwd = temp_workspace();
        let outside = temp_workspace();
        let escaped_dir = outside.join("escaped");
        fs::create_dir_all(&escaped_dir).unwrap();
        symlink(&escaped_dir, cwd.join("link")).unwrap();

        let registry = default_tool_registry(&cwd);
        let error = registry
            .execute(&call(
                "write",
                r#"{"path":"link/secret.txt","content":"nope"}"#,
            ))
            .unwrap_err();

        assert!(error.to_string().contains("path escapes workspace"));
        assert!(!escaped_dir.join("secret.txt").exists());

        let _ = fs::remove_dir_all(cwd);
        let _ = fs::remove_dir_all(outside);
    }

    #[test]
    fn bash_reports_failure_as_error_output() {
        let cwd = temp_workspace();
        let registry = default_tool_registry(&cwd);

        let output = registry
            .execute(&call("bash", r#"{"command":"echo boom >&2; exit 7"}"#))
            .unwrap();
        assert!(output.is_error);
        assert!(output.content.contains("boom"));
        assert!(output.content.contains("status"));

        let _ = fs::remove_dir_all(cwd);
    }

    #[cfg(unix)]
    #[test]
    fn bash_times_out_and_terminates_the_process_group() {
        let cwd = temp_workspace();
        let registry = default_tool_registry(&cwd);
        let started = Instant::now();

        let output = registry
            .execute(&call(
                "bash",
                r#"{"command":"sleep 10","timeout_seconds":1}"#,
            ))
            .unwrap();

        assert!(output.is_error);
        assert!(output.content.contains("timed out after 1s"));
        assert!(started.elapsed() < Duration::from_secs(4));

        let _ = fs::remove_dir_all(cwd);
    }

    #[test]
    fn bounded_reader_discards_bytes_beyond_its_memory_limit() {
        let input = std::io::Cursor::new(vec![b'x'; 128]);
        let output = join_reader(bounded_reader(input, 32), "test").unwrap();

        assert_eq!(output.bytes.len(), 32);
        assert!(output.truncated);
    }

    #[test]
    fn edit_requires_a_unique_non_empty_target() {
        let cwd = temp_workspace();
        fs::write(cwd.join("repeated.txt"), "same same").unwrap();
        let registry = default_tool_registry(&cwd);

        let repeated = registry
            .execute(&call(
                "edit",
                r#"{"path":"repeated.txt","old":"same","new":"changed"}"#,
            ))
            .unwrap_err();
        assert!(repeated.to_string().contains("more than once"));

        let empty = registry
            .execute(&call(
                "edit",
                r#"{"path":"repeated.txt","old":"","new":"changed"}"#,
            ))
            .unwrap_err();
        assert!(empty.to_string().contains("must not be empty"));

        let _ = fs::remove_dir_all(cwd);
    }

    #[test]
    fn read_rejects_files_that_would_overflow_the_model_context() {
        let cwd = temp_workspace();
        let path = cwd.join("large.txt");
        let file = fs::File::create(&path).unwrap();
        file.set_len(MAX_READ_BYTES + 1).unwrap();
        let registry = default_tool_registry(&cwd);

        let error = registry
            .execute(&call("read", r#"{"path":"large.txt"}"#))
            .unwrap_err();
        assert!(error.to_string().contains("refusing to read"));

        let _ = fs::remove_dir_all(cwd);
    }
}

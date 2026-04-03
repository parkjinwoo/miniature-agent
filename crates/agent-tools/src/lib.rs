use std::collections::HashMap;
use std::fs;
use std::path::Component;
use std::path::{Path, PathBuf};
use std::process::Command;

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
    tools: HashMap<String, Box<dyn Tool>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
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

pub fn default_tool_registry(cwd: impl Into<PathBuf>) -> ToolRegistry {
    let cwd = cwd.into();
    let mut registry = ToolRegistry::new();
    registry.register(ReadTool { cwd: cwd.clone() });
    registry.register(WriteTool { cwd: cwd.clone() });
    registry.register(EditTool { cwd: cwd.clone() });
    registry.register(BashTool { cwd });
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
            input_schema: serde_json::to_value(build_schema::<ReadArgs>()).unwrap_or(serde_json::json!({})),
        }
    }

    fn run(&self, call: &ToolCall) -> anyhow::Result<ToolOutput> {
        let args: ReadArgs = serde_json::from_str(&call.arguments_json)?;
        let path = resolve_path(&self.cwd, &args.path)?;
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
            input_schema: serde_json::to_value(build_schema::<WriteArgs>()).unwrap_or(serde_json::json!({})),
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
            input_schema: serde_json::to_value(build_schema::<EditArgs>()).unwrap_or(serde_json::json!({})),
        }
    }

    fn run(&self, call: &ToolCall) -> anyhow::Result<ToolOutput> {
        let args: EditArgs = serde_json::from_str(&call.arguments_json)?;
        let path = resolve_path(&self.cwd, &args.path)?;
        let content = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;

        if !content.contains(&args.old) {
            anyhow::bail!("target text not found in {}", path.display());
        }

        let updated = content.replacen(&args.old, &args.new, 1);
        fs::write(&path, updated)
            .with_context(|| format!("failed to write {}", path.display()))?;

        Ok(ToolOutput {
            content: format!("Edited {}", path.display()),
            is_error: false,
        })
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct BashArgs {
    command: String,
}

pub struct BashTool {
    cwd: PathBuf,
}

impl Tool for BashTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "bash".to_string(),
            description: "Run a shell command in the workspace".to_string(),
            input_schema: serde_json::to_value(build_schema::<BashArgs>()).unwrap_or(serde_json::json!({})),
        }
    }

    fn run(&self, call: &ToolCall) -> anyhow::Result<ToolOutput> {
        let args: BashArgs = serde_json::from_str(&call.arguments_json)?;
        let shell = std::env::var("SHELL")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "/bin/sh".to_string());
        let output = Command::new(&shell)
            .arg("-lc")
            .arg(&args.command)
            .current_dir(&self.cwd)
            .output()
            .with_context(|| format!("failed to run command with {}: {}", shell, args.command))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let mut content = String::new();
        if !stdout.is_empty() {
            content.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !content.is_empty() {
                content.push('\n');
            }
            content.push_str(&stderr);
        }
        if content.is_empty() {
            content = "(no output)".to_string();
        }

        Ok(ToolOutput {
            content,
            is_error: !output.status.success(),
        })
    }
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
    fn rejects_paths_outside_workspace() {
        let cwd = temp_workspace();
        let registry = default_tool_registry(&cwd);

        let error = registry
            .execute(&call("write", r#"{"path":"../escape.txt","content":"bad"}"#))
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

        let _ = fs::remove_dir_all(cwd);
    }
}

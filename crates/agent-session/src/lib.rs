use std::collections::HashMap;
use std::fs::{File, OpenOptions, create_dir_all, read_dir};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::{cmp, fmt::Write as _};

use agent_model::{LlmMessage, LlmRole, MessagePart, TextPart};
use agent_core::{AgentEvent, AgentMessage, AgentRunResult};
use anyhow::Context;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;
use uuid::Uuid;

pub const CURRENT_SESSION_VERSION: u32 = 1;
const DEFAULT_SUMMARY_ROLE: &str = "system";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionHeader {
    pub r#type: String,
    pub version: u32,
    pub id: String,
    pub cwd: String,
    pub created_at: String,
    pub parent_session: Option<String>,
    #[serde(default)]
    pub provider: Option<SessionProviderInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SessionProviderInfo {
    pub display_name: String,
    pub model: String,
    pub backend: String,
    pub resolved_base_url: Option<String>,
    pub compat: Option<SessionProviderCompat>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SessionProviderCompat {
    pub supports_reasoning_effort: bool,
    pub supports_developer_role: bool,
    pub requires_tool_result_name: bool,
    pub reasoning_field: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionEntryBase {
    pub id: String,
    pub parent_id: Option<String>,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MessageEntry {
    #[serde(flatten)]
    pub base: SessionEntryBase,
    pub r#type: String,
    pub message: AgentMessage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EventEntry {
    #[serde(flatten)]
    pub base: SessionEntryBase,
    pub r#type: String,
    pub event: AgentEvent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SummaryEntry {
    #[serde(flatten)]
    pub base: SessionEntryBase,
    pub r#type: String,
    pub original_leaf_id: String,
    pub compacted_message_count: usize,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum FileEntry {
    Header(SessionHeader),
    Message(MessageEntry),
    Event(EventEntry),
    Summary(SummaryEntry),
}

#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub path: PathBuf,
    pub id: String,
    pub cwd: String,
    pub created_at: String,
    pub parent_session: Option<String>,
    pub provider_display_name: Option<String>,
    pub provider_backend: Option<String>,
    pub provider_model: Option<String>,
    pub message_count: usize,
    pub summary_count: usize,
}

#[derive(Debug, Clone)]
pub struct SessionCheckpoint {
    pub entry_id: String,
    pub parent_id: Option<String>,
    pub timestamp: String,
    pub label: String,
    pub depth: usize,
    pub is_current_leaf: bool,
}

#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub summary_entry_id: String,
    pub leaf_id: Option<String>,
    pub compacted_message_count: usize,
}

pub struct SessionStore {
    path: PathBuf,
    header: SessionHeader,
    entries: Vec<FileEntry>,
    leaf_id: Option<String>,
}

impl SessionStore {
    pub fn create(
        session_dir: impl AsRef<Path>,
        cwd: impl Into<String>,
        parent_session: Option<String>,
        provider: Option<SessionProviderInfo>,
    ) -> anyhow::Result<Self> {
        let session_dir = session_dir.as_ref();
        create_dir_all(session_dir)
            .with_context(|| format!("failed to create session dir {}", session_dir.display()))?;

        let session_id = short_id();
        let path = session_dir.join(format!("{session_id}.jsonl"));
        let header = SessionHeader {
            r#type: "session".to_string(),
            version: CURRENT_SESSION_VERSION,
            id: session_id,
            cwd: cwd.into(),
            created_at: now_timestamp(),
            parent_session,
            provider,
        };

        let mut store = Self {
            path,
            header: header.clone(),
            entries: vec![FileEntry::Header(header)],
            leaf_id: None,
        };
        store.rewrite_file()?;
        Ok(store)
    }

    pub fn open_or_create_latest(
        session_dir: impl AsRef<Path>,
        cwd: impl Into<String>,
        provider: Option<SessionProviderInfo>,
    ) -> anyhow::Result<Self> {
        let session_dir = session_dir.as_ref();
        let cwd = cwd.into();

        if let Some(summary) = Self::list_sessions(session_dir)?
            .into_iter()
            .find(|session| session.cwd == cwd)
        {
            return Self::open(summary.path);
        }

        Self::create(session_dir, cwd, None, provider)
    }

    pub fn fork_latest(
        session_dir: impl AsRef<Path>,
        cwd: impl Into<String>,
    ) -> anyhow::Result<Self> {
        let session_dir = session_dir.as_ref();
        let cwd = cwd.into();

        if let Some(path) = Self::latest_session_path(session_dir)? {
            let store = Self::open(&path)?;
            if store.header.cwd == cwd {
                return store.fork(session_dir);
            }
        }

        Self::create(session_dir, cwd, None, None)
    }

    pub fn open(path: impl Into<PathBuf>) -> anyhow::Result<Self> {
        let path = path.into();
        let file = File::open(&path).with_context(|| format!("failed to open {}", path.display()))?;
        let reader = BufReader::new(file);

        let mut entries = Vec::new();
        let mut header = None;
        let mut leaf_id = None;

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let entry: FileEntry = serde_json::from_str(&line)
                .with_context(|| format!("failed to parse session line in {}", path.display()))?;

            match &entry {
                FileEntry::Header(found) => header = Some(found.clone()),
                FileEntry::Message(message) => leaf_id = Some(message.base.id.clone()),
                FileEntry::Event(event) => leaf_id = Some(event.base.id.clone()),
                FileEntry::Summary(summary) => leaf_id = Some(summary.base.id.clone()),
            }

            entries.push(entry);
        }

        Ok(Self {
            path,
            header: header.context("session file missing header")?,
            entries,
            leaf_id,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn header(&self) -> &SessionHeader {
        &self.header
    }

    pub fn leaf_id(&self) -> Option<&str> {
        self.leaf_id.as_deref()
    }

    pub fn set_leaf(&mut self, leaf_id: Option<String>) {
        self.leaf_id = leaf_id;
    }

    pub fn fork(&self, session_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        self.fork_with_provider(session_dir, self.header.provider.clone())
    }

    pub fn fork_with_provider(
        &self,
        session_dir: impl AsRef<Path>,
        provider: Option<SessionProviderInfo>,
    ) -> anyhow::Result<Self> {
        let mut forked = Self::create(
            session_dir,
            self.header.cwd.clone(),
            Some(self.header.id.clone()),
            provider,
        )?;

        for entry in &self.entries {
            match entry {
                FileEntry::Header(_) => {}
                FileEntry::Message(message) => {
                    let cloned = MessageEntry {
                        base: forked.next_base(),
                        r#type: "message".to_string(),
                        message: message.message.clone(),
                    };
                    forked.append_entry(FileEntry::Message(cloned))?;
                }
                FileEntry::Event(event) => {
                    let cloned = EventEntry {
                        base: forked.next_base(),
                        r#type: "event".to_string(),
                        event: event.event.clone(),
                    };
                    forked.append_entry(FileEntry::Event(cloned))?;
                }
                FileEntry::Summary(summary) => {
                    let cloned = SummaryEntry {
                        base: forked.next_base(),
                        r#type: "summary".to_string(),
                        original_leaf_id: summary.original_leaf_id.clone(),
                        compacted_message_count: summary.compacted_message_count,
                        summary: summary.summary.clone(),
                    };
                    forked.append_entry(FileEntry::Summary(cloned))?;
                }
            }
        }

        Ok(forked)
    }

    pub fn messages(&self) -> Vec<AgentMessage> {
        self.messages_for_leaf(self.leaf_id())
    }

    pub fn messages_for_leaf(&self, leaf_id: Option<&str>) -> Vec<AgentMessage> {
        let Some(leaf_id) = leaf_id else {
            return Vec::new();
        };

        let by_id = self.entry_index();
        let lineage = self.lineage_ids(leaf_id, &by_id);
        let mut messages = Vec::new();

        for id in lineage {
            match by_id.get(&id) {
                Some(EntryRef::Message(message)) => messages.push(message.message.clone()),
                Some(EntryRef::Summary(summary)) => {
                    messages.push(synthetic_summary_message(summary));
                }
                Some(EntryRef::Event(_)) | None => {}
            }
        }

        messages
    }

    pub fn checkpoints(&self) -> Vec<SessionCheckpoint> {
        let by_id = self.entry_index();
        let depth_by_id = self.depth_by_id(&by_id);

        self.entries
            .iter()
            .filter_map(|entry| match entry {
                FileEntry::Message(message) => Some(SessionCheckpoint {
                    entry_id: message.base.id.clone(),
                    parent_id: message.base.parent_id.clone(),
                    timestamp: message.base.timestamp.clone(),
                    label: checkpoint_label(&message.message),
                    depth: depth_by_id.get(&message.base.id).copied().unwrap_or(0),
                    is_current_leaf: self.leaf_id.as_deref() == Some(message.base.id.as_str()),
                }),
                FileEntry::Summary(summary) => Some(SessionCheckpoint {
                    entry_id: summary.base.id.clone(),
                    parent_id: summary.base.parent_id.clone(),
                    timestamp: summary.base.timestamp.clone(),
                    label: summary_label(summary),
                    depth: depth_by_id.get(&summary.base.id).copied().unwrap_or(0),
                    is_current_leaf: self.leaf_id.as_deref() == Some(summary.base.id.as_str()),
                }),
                FileEntry::Header(_) | FileEntry::Event(_) => None,
            })
            .collect()
    }

    pub fn compact_leaf(&mut self, keep_last: usize) -> anyhow::Result<Option<CompactionResult>> {
        let compacted_messages = self.compactable_messages(keep_last);
        if compacted_messages.is_empty() {
            return Ok(None);
        }
        let summary_text = build_compaction_summary(compacted_messages.into_iter());
        self.compact_leaf_with_summary(keep_last, summary_text)
    }

    pub fn compactable_messages(&self, keep_last: usize) -> Vec<AgentMessage> {
        let Some(leaf_id) = self.leaf_id.clone() else {
            return Vec::new();
        };

        let by_id = self.entry_index();
        let lineage = self.lineage_ids(&leaf_id, &by_id);
        let message_ids = lineage
            .into_iter()
            .filter(|id| matches!(by_id.get(id), Some(EntryRef::Message(_))))
            .collect::<Vec<_>>();

        if message_ids.len() <= keep_last.saturating_add(1) {
            return Vec::new();
        }

        let compact_count = message_ids.len().saturating_sub(keep_last);
        if compact_count == 0 {
            return Vec::new();
        }

        let compacted_ids = &message_ids[..compact_count];
        compacted_ids
            .iter()
            .filter_map(|id| by_id.get(id))
            .filter_map(|entry| match entry {
                EntryRef::Message(message) => Some(message.message.clone()),
                EntryRef::Summary(_) | EntryRef::Event(_) => None,
            })
            .collect()
    }

    pub fn compact_leaf_with_summary(
        &mut self,
        keep_last: usize,
        summary_text: String,
    ) -> anyhow::Result<Option<CompactionResult>> {
        let Some(leaf_id) = self.leaf_id.clone() else {
            return Ok(None);
        };

        let by_id = self.entry_index();
        let lineage = self.lineage_ids(&leaf_id, &by_id);
        let message_ids = lineage
            .into_iter()
            .filter(|id| matches!(by_id.get(id), Some(EntryRef::Message(_))))
            .collect::<Vec<_>>();

        if message_ids.len() <= keep_last.saturating_add(1) {
            return Ok(None);
        }

        let compact_count = message_ids.len().saturating_sub(keep_last);
        if compact_count == 0 {
            return Ok(None);
        }

        let compacted_ids = &message_ids[..compact_count];
        let kept_ids = &message_ids[compact_count..];

        let parent_id = compacted_ids
            .first()
            .and_then(|id| by_id.get(id))
            .and_then(EntryRef::parent_id)
            .map(ToString::to_string);

        let kept_messages = kept_ids
            .iter()
            .filter_map(|id| by_id.get(id))
            .filter_map(|entry| match entry {
                EntryRef::Message(message) => Some(message.message.clone()),
                EntryRef::Summary(_) | EntryRef::Event(_) => None,
            })
            .collect::<Vec<_>>();

        let summary_entry = SummaryEntry {
            base: SessionEntryBase {
                id: short_id(),
                parent_id,
                timestamp: now_timestamp(),
            },
            r#type: "summary".to_string(),
            original_leaf_id: leaf_id,
            compacted_message_count: compact_count,
            summary: summary_text,
        };
        let summary_id = summary_entry.base.id.clone();
        self.append_entry(FileEntry::Summary(summary_entry))?;

        for message in kept_messages {
            let cloned = MessageEntry {
                base: self.next_base(),
                r#type: "message".to_string(),
                message,
            };
            self.append_entry(FileEntry::Message(cloned))?;
        }

        Ok(Some(CompactionResult {
            summary_entry_id: summary_id,
            leaf_id: self.leaf_id.clone(),
            compacted_message_count: compact_count,
        }))
    }

    pub fn summary(&self) -> SessionSummary {
        SessionSummary {
            path: self.path.clone(),
            id: self.header.id.clone(),
            cwd: self.header.cwd.clone(),
            created_at: self.header.created_at.clone(),
            parent_session: self.header.parent_session.clone(),
            provider_display_name: self
                .header
                .provider
                .as_ref()
                .map(|provider| provider.display_name.clone()),
            provider_backend: self
                .header
                .provider
                .as_ref()
                .map(|provider| provider.backend.clone()),
            provider_model: self
                .header
                .provider
                .as_ref()
                .map(|provider| provider.model.clone()),
            message_count: self
                .entries
                .iter()
                .filter(|entry| matches!(entry, FileEntry::Message(_)))
                .count(),
            summary_count: self
                .entries
                .iter()
                .filter(|entry| matches!(entry, FileEntry::Summary(_)))
                .count(),
        }
    }

    pub fn list_sessions(session_dir: impl AsRef<Path>) -> anyhow::Result<Vec<SessionSummary>> {
        let session_dir = session_dir.as_ref();
        if !session_dir.exists() {
            return Ok(Vec::new());
        }

        let mut sessions = Vec::new();
        for entry in read_dir(session_dir)
            .with_context(|| format!("failed to read session dir {}", session_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
                continue;
            }
            if let Ok(store) = Self::open(&path) {
                sessions.push(store.summary());
            }
        }

        sessions.sort_by(|a, b| {
            b.created_at
                .cmp(&a.created_at)
                .then_with(|| b.path.cmp(&a.path))
        });

        Ok(sessions)
    }

    pub fn latest_session_path(session_dir: impl AsRef<Path>) -> anyhow::Result<Option<PathBuf>> {
        Ok(Self::list_sessions(session_dir)?
            .into_iter()
            .next()
            .map(|session| session.path))
    }

    pub fn append_run(&mut self, run: &AgentRunResult) -> anyhow::Result<()> {
        for message in &run.new_messages {
            let entry = MessageEntry {
                base: self.next_base(),
                r#type: "message".to_string(),
                message: message.clone(),
            };
            self.append_entry(FileEntry::Message(entry))?;
        }

        for event in &run.events {
            let entry = EventEntry {
                base: self.next_base(),
                r#type: "event".to_string(),
                event: event.clone(),
            };
            self.append_entry(FileEntry::Event(entry))?;
        }

        Ok(())
    }

    fn next_base(&self) -> SessionEntryBase {
        SessionEntryBase {
            id: short_id(),
            parent_id: self.leaf_id.clone(),
            timestamp: now_timestamp(),
        }
    }

    fn append_entry(&mut self, entry: FileEntry) -> anyhow::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("failed to append {}", self.path.display()))?;

        serde_json::to_writer(&mut file, &entry)?;
        file.write_all(b"\n")?;
        file.flush()?;

        match &entry {
            FileEntry::Header(_) => {}
            FileEntry::Message(message) => self.leaf_id = Some(message.base.id.clone()),
            FileEntry::Event(event) => self.leaf_id = Some(event.base.id.clone()),
            FileEntry::Summary(summary) => self.leaf_id = Some(summary.base.id.clone()),
        }

        self.entries.push(entry);
        Ok(())
    }

    fn rewrite_file(&mut self) -> anyhow::Result<()> {
        let mut file = File::create(&self.path)
            .with_context(|| format!("failed to write {}", self.path.display()))?;

        for entry in &self.entries {
            serde_json::to_writer(&mut file, entry)?;
            file.write_all(b"\n")?;
        }
        file.flush()?;
        Ok(())
    }

    fn entry_index(&self) -> HashMap<String, EntryRef<'_>> {
        let mut by_id = HashMap::new();
        for entry in &self.entries {
            match entry {
                FileEntry::Header(_) => {}
                FileEntry::Message(message) => {
                    by_id.insert(message.base.id.clone(), EntryRef::Message(message));
                }
                FileEntry::Event(event) => {
                    by_id.insert(event.base.id.clone(), EntryRef::Event(event));
                }
                FileEntry::Summary(summary) => {
                    by_id.insert(summary.base.id.clone(), EntryRef::Summary(summary));
                }
            }
        }
        by_id
    }

    fn lineage_ids(&self, leaf_id: &str, by_id: &HashMap<String, EntryRef<'_>>) -> Vec<String> {
        let mut lineage = Vec::new();
        let mut current = Some(leaf_id.to_string());

        while let Some(id) = current {
            let Some(entry) = by_id.get(&id) else {
                break;
            };
            lineage.push(id.clone());
            current = entry.parent_id().map(ToString::to_string);
        }

        lineage.reverse();
        lineage
    }

    fn depth_by_id(&self, by_id: &HashMap<String, EntryRef<'_>>) -> HashMap<String, usize> {
        let mut depth_by_id = HashMap::new();

        for entry in &self.entries {
            let (id, parent_id) = match entry {
                FileEntry::Header(_) => continue,
                FileEntry::Message(message) => (&message.base.id, message.base.parent_id.as_deref()),
                FileEntry::Event(event) => (&event.base.id, event.base.parent_id.as_deref()),
                FileEntry::Summary(summary) => (&summary.base.id, summary.base.parent_id.as_deref()),
            };

            let parent_depth = parent_id
                .and_then(|parent| depth_by_id.get(parent).copied())
                .or_else(|| {
                    parent_id
                        .and_then(|parent| by_id.get(parent))
                        .map(|_| 0usize)
                })
                .unwrap_or(0);
            depth_by_id.insert(id.clone(), parent_depth + 1);
        }

        depth_by_id
    }
}

fn short_id() -> String {
    Uuid::new_v4().to_string()[..8].to_string()
}

fn now_timestamp() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string())
}

enum EntryRef<'a> {
    Message(&'a MessageEntry),
    Event(&'a EventEntry),
    Summary(&'a SummaryEntry),
}

impl<'a> EntryRef<'a> {
    fn parent_id(&self) -> Option<&str> {
        match self {
            EntryRef::Message(message) => message.base.parent_id.as_deref(),
            EntryRef::Event(event) => event.base.parent_id.as_deref(),
            EntryRef::Summary(summary) => summary.base.parent_id.as_deref(),
        }
    }
}

fn synthetic_summary_message(summary: &SummaryEntry) -> AgentMessage {
    AgentMessage::User(LlmMessage {
        role: LlmRole::User,
        parts: vec![MessagePart::Text(TextPart {
            text: format!(
                "[{}] {} messages compacted.\n{}",
                DEFAULT_SUMMARY_ROLE,
                summary.compacted_message_count,
                summary.summary
            ),
        })],
    })
}

fn checkpoint_label(message: &AgentMessage) -> String {
    let llm = message.as_llm_message();
    let role = match message {
        AgentMessage::User(_) => "You",
        AgentMessage::Assistant(_) => "AI",
        AgentMessage::ToolResult(_) => "Tool",
    };

    let mut summary = String::new();
    for part in &llm.parts {
        match part {
            MessagePart::Text(text) => {
                let snippet = text.text.replace('\n', " ");
                let _ = write!(summary, "{snippet}");
                break;
            }
            MessagePart::ToolCall(call) => {
                let _ = write!(summary, "tool_call {}", call.name);
                break;
            }
            MessagePart::ToolResult(result) => {
                let _ = write!(summary, "tool_result {}", result.call_id);
                break;
            }
        }
    }

    let snippet = truncate_label(&summary, 160);
    format!("{role}: {snippet}")
}

fn summary_label(summary: &SummaryEntry) -> String {
    let prefix = truncate_label(&summary.summary.replace('\n', " "), 160);
    format!("Summary: {} compacted ({prefix})", summary.compacted_message_count)
}

fn truncate_label(input: &str, max_chars: usize) -> String {
    let chars = input.chars().collect::<Vec<_>>();
    if chars.len() <= max_chars {
        return input.to_string();
    }

        let keep = cmp::max(0, max_chars.saturating_sub(3));
        let mut out = chars.into_iter().take(keep).collect::<String>();
        out.push_str("...");
        out
}

fn build_compaction_summary(messages: impl Iterator<Item = AgentMessage>) -> String {
    let mut lines = Vec::new();

    for message in messages {
        let role = match &message {
            AgentMessage::User(_) => "user",
            AgentMessage::Assistant(_) => "assistant",
            AgentMessage::ToolResult(_) => "tool",
        };

        let mut body = String::new();
        for part in &message.as_llm_message().parts {
            match part {
                MessagePart::Text(text) => {
                    if !body.is_empty() {
                        body.push(' ');
                    }
                    body.push_str(text.text.trim());
                }
                MessagePart::ToolCall(call) => {
                    if !body.is_empty() {
                        body.push(' ');
                    }
                    let _ = write!(body, "tool_call {} {}", call.name, call.arguments_json);
                }
                MessagePart::ToolResult(result) => {
                    if !body.is_empty() {
                        body.push(' ');
                    }
                    let _ = write!(body, "tool_result {}", result.call_id);
                    if !result.content.is_empty() {
                        body.push(' ');
                        body.push_str(result.content.trim());
                    }
                }
            }
        }

        if body.is_empty() {
            body.push_str("(empty)");
        }

        lines.push(format!("- {role}: {}", truncate_label(&body.replace('\n', " "), 160)));
    }

    if lines.is_empty() {
        "No earlier messages were available to compact.".to_string()
    } else {
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    fn temp_session_dir() -> PathBuf {
        let unique = Uuid::new_v4();
        let path = std::env::temp_dir().join(format!("miniature-agent-session-{unique}"));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn text_message(role: LlmRole, text: &str) -> AgentMessage {
        let message = LlmMessage {
            role: role.clone(),
            parts: vec![MessagePart::Text(TextPart {
                text: text.to_string(),
            })],
        };
        match role {
            LlmRole::User => AgentMessage::User(message),
            LlmRole::Assistant => AgentMessage::Assistant(message),
            LlmRole::Tool | LlmRole::System => AgentMessage::ToolResult(message),
        }
    }

    fn provider(name: &str, model: &str) -> SessionProviderInfo {
        SessionProviderInfo {
            display_name: name.to_string(),
            model: model.to_string(),
            backend: format!("chat-completions:{}", name.to_lowercase()),
            resolved_base_url: Some("http://localhost".to_string()),
            compat: None,
        }
    }

    #[test]
    fn fork_with_provider_rewrites_snapshot_but_keeps_transcript() {
        let session_dir = temp_session_dir();
        let mut store = SessionStore::create(
            &session_dir,
            "/workspace",
            None,
            Some(provider("OpenAI", "gpt-5")),
        )
        .unwrap();
        store
            .append_run(&AgentRunResult {
                new_messages: vec![
                    text_message(LlmRole::User, "hello"),
                    text_message(LlmRole::Assistant, "world"),
                ],
                ..AgentRunResult::default()
            })
            .unwrap();

        let forked = store
            .fork_with_provider(&session_dir, Some(provider("Ollama", "gpt-oss:20b")))
            .unwrap();

        assert_eq!(
            forked.header().provider.as_ref().unwrap().display_name,
            "Ollama"
        );
        assert_eq!(
            forked.header().parent_session.as_deref(),
            Some(store.header().id.as_str())
        );
        assert_eq!(forked.messages(), store.messages());

        let _ = fs::remove_dir_all(session_dir);
    }

    #[test]
    fn compaction_inserts_summary_and_preserves_recent_messages() {
        let session_dir = temp_session_dir();
        let mut store = SessionStore::create(&session_dir, "/workspace", None, None).unwrap();
        store
            .append_run(&AgentRunResult {
                new_messages: vec![
                    text_message(LlmRole::User, "u1"),
                    text_message(LlmRole::Assistant, "a1"),
                    text_message(LlmRole::User, "u2"),
                    text_message(LlmRole::Assistant, "a2"),
                ],
                ..AgentRunResult::default()
            })
            .unwrap();

        let result = store
            .compact_leaf_with_summary(2, "summary of earlier messages".to_string())
            .unwrap()
            .unwrap();

        assert_eq!(result.compacted_message_count, 2);
        assert_eq!(store.summary().summary_count, 1);

        let restored = store.messages();
        assert_eq!(restored.len(), 3);
        let summary_text = match &restored[0].as_llm_message().parts[0] {
            MessagePart::Text(part) => part.text.clone(),
            _ => panic!("expected synthetic summary text"),
        };
        assert!(summary_text.contains("summary of earlier messages"));
        assert!(matches!(&restored[1], AgentMessage::User(_)));
        assert!(matches!(&restored[2], AgentMessage::Assistant(_)));

        let _ = fs::remove_dir_all(session_dir);
    }

    #[test]
    fn open_or_create_latest_prefers_matching_cwd_session() {
        let session_dir = temp_session_dir();
        let first = SessionStore::create(
            &session_dir,
            "/workspace-a",
            None,
            Some(provider("OpenAI", "gpt-5")),
        )
        .unwrap();
        let second = SessionStore::create(
            &session_dir,
            "/workspace-b",
            None,
            Some(provider("Ollama", "gpt-oss:20b")),
        )
        .unwrap();

        let reopened = SessionStore::open_or_create_latest(
            &session_dir,
            "/workspace-a",
            Some(provider("OpenAI", "gpt-5")),
        )
        .unwrap();

        assert_eq!(reopened.header().id, first.header().id);
        assert_ne!(reopened.header().id, second.header().id);

        let _ = fs::remove_dir_all(session_dir);
    }

    #[test]
    fn list_sessions_ignores_corrupted_files() {
        let session_dir = temp_session_dir();
        let valid = SessionStore::create(&session_dir, "/workspace", None, None).unwrap();
        fs::write(session_dir.join("broken.jsonl"), "{not json\n").unwrap();

        let sessions = SessionStore::list_sessions(&session_dir).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, valid.header().id);

        let _ = fs::remove_dir_all(session_dir);
    }
}

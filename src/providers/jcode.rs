//! jcode provider — reads/writes native jcode sessions.
//!
//! Storage layout (mirrors `jcode-storage` / `jcode-base`):
//! ```text
//! <jcode_dir>/sessions/<id>.json            ← snapshot (full Session)
//! <jcode_dir>/sessions/<id>.journal.jsonl   ← append-only log (applied on read)
//! ```
//! `jcode_dir` precedence: `$JCODE_HOME` > `$JCODE_USE_XDG` (XDG_DATA_HOME) >
//! `~/.jcode`.
//!
//! This provider is **self-contained**: it parses/serializes the jcode wire
//! format with local logic and does not depend on any jcode crate. jcode's
//! `Role` has only `user`/`assistant`, and tool results are stored as `user`
//! messages containing a `tool_result` content block.
//!
//! Resume command: `jcode --resume <id>`.

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde_json::{Value, json};
use tracing::{debug, info, trace};

use crate::discovery::DetectionResult;
use crate::model::{
    CanonicalMessage, CanonicalSession, MessageRole, ToolCall, ToolResult, flatten_content,
    normalize_role, parse_timestamp, reindex_messages, truncate_title,
};
use crate::providers::{Provider, WriteOptions, WrittenSession};

/// jcode provider implementation.
pub struct JCode;

impl JCode {
    /// Whether the user opted into XDG paths via `JCODE_USE_XDG`.
    fn xdg_enabled() -> bool {
        matches!(
            std::env::var("JCODE_USE_XDG")
                .ok()
                .as_deref()
                .map(str::trim)
                .map(str::to_ascii_lowercase)
                .as_deref(),
            Some("1") | Some("true") | Some("yes") | Some("on")
        )
    }

    /// Resolve the jcode home directory, matching `jcode-storage::jcode_dir`.
    fn jcode_dir() -> Option<PathBuf> {
        if let Ok(home) = std::env::var("JCODE_HOME") {
            return Some(PathBuf::from(home));
        }
        if Self::xdg_enabled() {
            return std::env::var("XDG_DATA_HOME")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .map(PathBuf::from)
                .or_else(|| dirs::home_dir().map(|h| h.join(".local").join("share")))
                .map(|x| x.join("jcode"));
        }
        dirs::home_dir().map(|h| h.join(".jcode"))
    }

    /// Directory containing session snapshots.
    fn sessions_dir() -> Option<PathBuf> {
        Self::jcode_dir().map(|d| d.join("sessions"))
    }

    /// Path to the journal file for a given snapshot path (`<stem>.journal.jsonl`).
    fn journal_path(snapshot: &Path) -> PathBuf {
        let mut name = snapshot.file_stem().unwrap_or_default().to_os_string();
        name.push(".journal.jsonl");
        snapshot.with_file_name(name)
    }
}

impl Provider for JCode {
    fn name(&self) -> &str {
        "jcode"
    }

    fn slug(&self) -> &str {
        "jcode"
    }

    fn cli_alias(&self) -> &str {
        "jc"
    }

    fn detect(&self) -> DetectionResult {
        let mut evidence = Vec::new();
        let mut installed = false;

        if which::which("jcode").is_ok() {
            evidence.push("jcode binary found in PATH".to_string());
            installed = true;
        }
        if let Some(dir) = Self::sessions_dir()
            && dir.is_dir()
        {
            evidence.push(format!("{} exists", dir.display()));
            installed = true;
        }

        trace!(provider = "jcode", ?evidence, installed, "detection");
        DetectionResult {
            installed,
            version: None,
            evidence,
        }
    }

    fn session_roots(&self) -> Vec<PathBuf> {
        match Self::sessions_dir() {
            Some(dir) if dir.is_dir() => vec![dir],
            _ => vec![],
        }
    }

    fn list_sessions(&self) -> Option<Vec<(String, PathBuf)>> {
        let dir = Self::sessions_dir()?;
        if !dir.is_dir() {
            return Some(vec![]);
        }
        let mut sessions = Vec::new();
        for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            // Snapshots end in `.json` but not `.journal.jsonl`.
            if !name.ends_with(".json") || name.ends_with(".journal.jsonl") {
                continue;
            }
            if let Some(id) = name.strip_suffix(".json") {
                sessions.push((id.to_string(), path.clone()));
            }
        }
        Some(sessions)
    }

    fn owns_session(&self, session_id: &str) -> Option<PathBuf> {
        let path = Self::sessions_dir()?.join(format!("{session_id}.json"));
        path.is_file().then_some(path)
    }

    fn read_session(&self, path: &Path) -> anyhow::Result<CanonicalSession> {
        debug!(path = %path.display(), "reading jcode session");

        let file = std::fs::File::open(path)
            .with_context(|| format!("failed to open {}", path.display()))?;
        let root: Value = serde_json::from_reader(std::io::BufReader::new(file))
            .with_context(|| format!("failed to parse JSON {}", path.display()))?;

        // Snapshot messages, then apply journal append-entries (no incremental replay).
        let mut msgs_json: Vec<Value> = root
            .get("messages")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let journal = Self::journal_path(path);
        if journal.is_file()
            && let Ok(text) = std::fs::read_to_string(&journal)
        {
            for line in text.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let Ok(entry) = serde_json::from_str::<Value>(trimmed) else {
                    break;
                };
                if let Some(appended) = entry.get("append_messages").and_then(|v| v.as_array()) {
                    msgs_json.extend(appended.iter().cloned());
                }
            }
        }

        let session_id = root
            .get("id")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string()
            });

        let started_at = root.get("created_at").and_then(parse_timestamp);
        let mut ended_at = root.get("updated_at").and_then(parse_timestamp);

        let mut messages: Vec<CanonicalMessage> = Vec::with_capacity(msgs_json.len());
        for msg in &msgs_json {
            let role = normalize_role(msg.get("role").and_then(|v| v.as_str()).unwrap_or("user"));
            let (content, tool_calls, tool_results) = blocks_to_canonical(msg.get("content"));
            let timestamp = msg.get("timestamp").and_then(parse_timestamp);
            if let Some(t) = timestamp {
                ended_at = Some(ended_at.map_or(t, |e: i64| e.max(t)));
            }
            messages.push(CanonicalMessage {
                idx: 0,
                role,
                content,
                timestamp,
                author: None,
                tool_calls,
                tool_results,
                extra: msg.clone(),
            });
        }
        reindex_messages(&mut messages);

        let title = root
            .get("title")
            .and_then(|v| v.as_str())
            .map(ToString::to_string)
            .or_else(|| {
                messages
                    .iter()
                    .find(|m| m.role == MessageRole::User)
                    .map(|m| truncate_title(&m.content, 100))
            });

        let workspace = root
            .get("working_dir")
            .and_then(|v| v.as_str())
            .map(PathBuf::from);
        let model_name = root
            .get("model")
            .and_then(|v| v.as_str())
            .map(ToString::to_string);

        // Preserve jcode-specific fields for round-trip fidelity.
        let mut jcode_meta = serde_json::Map::new();
        for key in [
            "provider_session_id",
            "provider_key",
            "model",
            "reasoning_effort",
            "status",
            "compaction",
        ] {
            if let Some(v) = root.get(key) {
                jcode_meta.insert(key.to_string(), v.clone());
            }
        }
        let metadata = json!({ "source": "jcode", "jcode": Value::Object(jcode_meta) });

        debug!(
            session_id,
            messages = messages.len(),
            "jcode session parsed"
        );
        Ok(CanonicalSession {
            session_id,
            provider_slug: "jcode".to_string(),
            workspace,
            title,
            started_at,
            ended_at,
            messages,
            metadata,
            source_path: path.to_path_buf(),
            model_name,
        })
    }

    fn write_session(
        &self,
        session: &CanonicalSession,
        opts: &WriteOptions,
    ) -> anyhow::Result<WrittenSession> {
        let target_session_id = opts
            .target_session_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let now = chrono::Utc::now();
        let target_path = Self::sessions_dir()
            .ok_or_else(|| anyhow::anyhow!("cannot determine jcode sessions directory"))?
            .join(format!("{target_session_id}.json"));

        let snapshot = build_snapshot(session, &target_session_id, now);
        let bytes = serde_json::to_string_pretty(&snapshot)?.into_bytes();
        let outcome = crate::pipeline::atomic_write(&target_path, &bytes, opts.force, self.slug())?;

        info!(
            target_session_id,
            path = %outcome.target_path.display(),
            messages = session.messages.len(),
            "jcode session written"
        );
        Ok(WrittenSession {
            paths: vec![outcome.target_path],
            session_id: target_session_id.clone(),
            resume_command: self.resume_command(&target_session_id),
            backup_path: outcome.backup_path,
        })
    }

    fn resume_command(&self, session_id: &str) -> String {
        format!("jcode --resume {session_id}")
    }
}

// ---------------------------------------------------------------------------
// Block <-> canonical conversion
// ---------------------------------------------------------------------------

fn jcode_role(role: &MessageRole) -> &'static str {
    // jcode only has user/assistant; everything non-assistant collapses to user.
    match role {
        MessageRole::Assistant => "assistant",
        _ => "user",
    }
}

/// Parse a jcode `content` block array into (text, tool_calls, tool_results).
fn blocks_to_canonical(content: Option<&Value>) -> (String, Vec<ToolCall>, Vec<ToolResult>) {
    let mut parts: Vec<String> = Vec::new();
    let mut calls: Vec<ToolCall> = Vec::new();
    let mut results: Vec<ToolResult> = Vec::new();

    let Some(blocks) = content.and_then(|v| v.as_array()) else {
        // Tolerate a plain string content.
        if let Some(s) = content.and_then(|v| v.as_str()) {
            return (s.to_string(), calls, results);
        }
        return (String::new(), calls, results);
    };

    for block in blocks {
        match block.get("type").and_then(|v| v.as_str()) {
            Some("text") | Some("reasoning") => {
                if let Some(s) = block.get("text").and_then(|v| v.as_str()) {
                    parts.push(s.to_string());
                }
            }
            Some("anthropic_thinking") => {
                if let Some(s) = block.get("thinking").and_then(|v| v.as_str()) {
                    parts.push(s.to_string());
                }
            }
            Some("openai_reasoning") => {
                if let Some(arr) = block.get("summary").and_then(|v| v.as_array()) {
                    let joined: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
                    if !joined.is_empty() {
                        parts.push(joined.join("\n"));
                    }
                }
            }
            Some("image") => {
                let mt = block
                    .get("media_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                parts.push(format!("[Image: {mt}]"));
            }
            Some("tool_use") => calls.push(ToolCall {
                id: block
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(ToString::to_string),
                name: block
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                arguments: block.get("input").cloned().unwrap_or(Value::Null),
            }),
            Some("tool_result") => results.push(ToolResult {
                call_id: block
                    .get("tool_use_id")
                    .and_then(|v| v.as_str())
                    .map(ToString::to_string),
                content: block.get("content").and_then(|v| v.as_str()).map_or_else(
                    || {
                        block
                            .get("content")
                            .map(flatten_content)
                            .unwrap_or_default()
                    },
                    ToString::to_string,
                ),
                is_error: block
                    .get("is_error")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
            }),
            // `openai_compaction` (runtime-only) and unknown blocks are skipped.
            _ => {}
        }
    }

    (parts.join("\n"), calls, results)
}

/// Build a jcode `content` block array from a canonical message.
///
/// Emits a single text block (when content is non-empty) plus reconstructed
/// `tool_use` / `tool_result` blocks. This is the exact inverse of
/// [`blocks_to_canonical`] for text + tool content, so the pipeline's
/// read-back verification (count + role bucket + content) holds.
fn canonical_to_blocks(msg: &CanonicalMessage) -> Vec<Value> {
    let mut blocks: Vec<Value> = Vec::new();
    if !msg.content.is_empty() {
        blocks.push(json!({ "type": "text", "text": msg.content }));
    }
    for tc in &msg.tool_calls {
        blocks.push(json!({
            "type": "tool_use",
            "id": tc.id.clone().unwrap_or_default(),
            "name": tc.name,
            "input": tc.arguments,
        }));
    }
    for tr in &msg.tool_results {
        let mut block = json!({
            "type": "tool_result",
            "tool_use_id": tr.call_id.clone().unwrap_or_default(),
            "content": tr.content,
        });
        if tr.is_error {
            block["is_error"] = Value::Bool(true);
        }
        blocks.push(block);
    }
    blocks
}

/// Build a jcode snapshot JSON object from a canonical session (no IO).
///
/// Imported sessions start a fresh provider thread: `provider_session_id` /
/// `provider_key` / `model` are intentionally omitted so jcode does not try to
/// resume another provider's backend or restore an unavailable model. `status`
/// defaults to `Active`.
fn build_snapshot(
    session: &CanonicalSession,
    session_id: &str,
    now: chrono::DateTime<chrono::Utc>,
) -> Value {
    let rfc3339 = |ms: Option<i64>| {
        ms.and_then(chrono::DateTime::from_timestamp_millis)
            .unwrap_or(now)
            .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
    };

    let json_messages: Vec<Value> = session
        .messages
        .iter()
        .map(|msg| {
            let mut entry = json!({
                "id": format!("m{}", msg.idx),
                "role": jcode_role(&msg.role),
                "content": canonical_to_blocks(msg),
            });
            if let Some(ts) = msg
                .timestamp
                .and_then(chrono::DateTime::from_timestamp_millis)
                .map(|dt| dt.to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
            {
                entry["timestamp"] = Value::String(ts);
            }
            entry
        })
        .collect();

    let mut snapshot = json!({
        "id": session_id,
        "created_at": rfc3339(session.started_at),
        "updated_at": rfc3339(session.ended_at),
        "messages": json_messages,
        "status": "Active",
    });
    if let Some(ws) = session.workspace.as_ref() {
        snapshot["working_dir"] = Value::String(ws.to_string_lossy().into_owned());
    }
    snapshot
}

#[cfg(test)]
mod tests {
    use super::{JCode, blocks_to_canonical, build_snapshot, canonical_to_blocks, jcode_role};
    use crate::model::{CanonicalMessage, CanonicalSession, MessageRole, ToolCall, ToolResult};
    use crate::providers::Provider;
    use std::io::Write as _;
    use std::path::PathBuf;

    fn read_jcode_json(content: &str) -> CanonicalSession {
        let mut tmp = tempfile::NamedTempFile::with_suffix(".json").unwrap();
        tmp.write_all(content.as_bytes()).unwrap();
        tmp.flush().unwrap();
        JCode
            .read_session(tmp.path())
            .unwrap_or_else(|e| panic!("read_session failed: {e}"))
    }

    fn msg(role: MessageRole, content: &str) -> CanonicalMessage {
        CanonicalMessage {
            idx: 0,
            role,
            content: content.to_string(),
            timestamp: Some(1_700_000_000_000),
            author: None,
            tool_calls: vec![],
            tool_results: vec![],
            extra: serde_json::Value::Null,
        }
    }

    #[test]
    fn resume_command_uses_resume_flag() {
        assert_eq!(JCode.resume_command("abc"), "jcode --resume abc");
    }

    #[test]
    fn role_collapses_non_assistant_to_user() {
        assert_eq!(jcode_role(&MessageRole::Assistant), "assistant");
        assert_eq!(jcode_role(&MessageRole::User), "user");
        assert_eq!(jcode_role(&MessageRole::Tool), "user");
        assert_eq!(jcode_role(&MessageRole::System), "user");
        assert_eq!(jcode_role(&MessageRole::Other("x".into())), "user");
    }

    #[test]
    fn reader_basic_exchange_and_metadata() {
        let session = read_jcode_json(
            r#"{
                "id": "jc-1",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:05:00Z",
                "working_dir": "/data/projects/foo",
                "model": "gpt-5",
                "provider_key": "openai",
                "messages": [
                    {"id":"m0","role":"user","content":[{"type":"text","text":"Hello"}]},
                    {"id":"m1","role":"assistant","content":[{"type":"text","text":"Hi"}]}
                ]
            }"#,
        );
        assert_eq!(session.session_id, "jc-1");
        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[0].role, MessageRole::User);
        assert_eq!(session.messages[1].role, MessageRole::Assistant);
        assert_eq!(session.workspace, Some(PathBuf::from("/data/projects/foo")));
        assert_eq!(session.model_name.as_deref(), Some("gpt-5"));
        assert_eq!(session.metadata["jcode"]["provider_key"], "openai");
    }

    #[test]
    fn reader_extracts_tool_blocks_and_skips_compaction() {
        let session = read_jcode_json(
            r#"{
                "id": "jc-tools",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "messages": [
                    {"id":"m0","role":"user","content":[{"type":"text","text":"go"}]},
                    {"id":"m1","role":"assistant","content":[
                        {"type":"text","text":"running"},
                        {"type":"tool_use","id":"c1","name":"Bash","input":{"cmd":"ls"}},
                        {"type":"openai_compaction","encrypted_content":"xxx"}
                    ]},
                    {"id":"m2","role":"user","content":[
                        {"type":"tool_result","tool_use_id":"c1","content":"ok"}
                    ]}
                ]
            }"#,
        );
        assert_eq!(session.messages.len(), 3);
        assert_eq!(session.messages[1].content, "running");
        assert_eq!(session.messages[1].tool_calls.len(), 1);
        assert_eq!(session.messages[1].tool_calls[0].name, "Bash");
        // compaction block produced no content / tools.
        assert!(session.messages[1].tool_results.is_empty());
        assert_eq!(session.messages[2].tool_results[0].content, "ok");
        assert!(session.messages[2].content.is_empty());
    }

    #[test]
    fn reader_applies_journal_appends() {
        let dir = tempfile::tempdir().unwrap();
        let snap = dir.path().join("jc-journal.json");
        std::fs::write(
            &snap,
            r#"{"id":"jc-journal","created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z","messages":[{"id":"m0","role":"user","content":[{"type":"text","text":"first"}]}]}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("jc-journal.journal.jsonl"),
            "{\"meta\":{},\"append_messages\":[{\"id\":\"m1\",\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"second\"}]}]}\n",
        )
        .unwrap();

        let session = JCode.read_session(&snap).unwrap();
        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[1].content, "second");
        assert_eq!(session.messages[1].role, MessageRole::Assistant);
    }

    #[test]
    fn blocks_round_trip_text_and_tools() {
        let mut m = msg(MessageRole::Assistant, "[Tool: Bash]\n[Tool Output] ok");
        m.tool_calls.push(ToolCall {
            id: Some("c1".into()),
            name: "Bash".into(),
            arguments: serde_json::json!({"cmd": "ls"}),
        });
        m.tool_results.push(ToolResult {
            call_id: Some("c1".into()),
            content: "ok".into(),
            is_error: false,
        });
        let blocks = serde_json::Value::Array(canonical_to_blocks(&m));
        let (content, calls, results) = blocks_to_canonical(Some(&blocks));
        assert_eq!(content, "[Tool: Bash]\n[Tool Output] ok");
        assert_eq!(calls.len(), 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "ok");
    }

    #[test]
    fn write_then_read_preserves_count_roles_content() {
        // Exercise the exact-inverse property the pipeline's read-back relies on,
        // via the pure `build_snapshot` builder (no IO/env needed).
        let mut tool_msg = msg(MessageRole::Tool, "[Tool Output] done");
        tool_msg.tool_results.push(ToolResult {
            call_id: Some("c1".into()),
            content: "done".into(),
            is_error: false,
        });
        let canonical = CanonicalSession {
            session_id: "src".into(),
            provider_slug: "codex".into(),
            workspace: Some(PathBuf::from("/data/projects/foo")),
            title: Some("t".into()),
            started_at: Some(1_700_000_000_000),
            ended_at: Some(1_700_000_010_000),
            messages: vec![
                msg(MessageRole::User, "do it"),
                msg(MessageRole::Assistant, "on it"),
                tool_msg,
            ],
            metadata: serde_json::Value::Null,
            source_path: PathBuf::from("/tmp/src"),
            model_name: Some("gpt-5-codex".into()),
        };

        let snapshot = build_snapshot(&canonical, "jc-new", chrono::Utc::now());
        // Imported snapshots must not carry a provider thread id / key / model.
        assert!(snapshot.get("provider_session_id").is_none());
        assert!(snapshot.get("provider_key").is_none());
        assert!(snapshot.get("model").is_none());

        let mut tmp = tempfile::NamedTempFile::with_suffix(".json").unwrap();
        tmp.write_all(serde_json::to_string(&snapshot).unwrap().as_bytes())
            .unwrap();
        tmp.flush().unwrap();
        let read = JCode.read_session(tmp.path()).unwrap();

        assert_eq!(read.messages.len(), canonical.messages.len());
        for (orig, rb) in canonical.messages.iter().zip(read.messages.iter()) {
            assert_eq!(orig.content, rb.content, "content must round-trip exactly");
            let is_assistant = |r: &MessageRole| matches!(r, MessageRole::Assistant);
            assert_eq!(
                is_assistant(&orig.role),
                is_assistant(&rb.role),
                "role bucket must match"
            );
        }
    }
}

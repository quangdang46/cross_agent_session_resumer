//! Grok Build (xAI) provider — reads/writes native Grok sessions under `~/.grok`.
//!
//! ## Storage layout
//!
//! ```text
//! <GROK_HOME|~/.grok>/sessions/<urlencoded-cwd>/<session-id>/
//!   summary.json         # metadata: title, timestamps, model, cwd
//!   updates.jsonl        # ACP session/update stream (authoritative for resume)
//!   chat_history.jsonl   # raw model chat messages (user/assistant/tool_result)
//! ```
//!
//! Working directories are URL-encoded (e.g. `/Users/me/proj` →
//! `%2FUsers%2Fme%2Fproj`). When the encoded name would exceed 255 bytes Grok
//! uses a slug+hash directory with a `.cwd` sidecar; casr reads `.cwd` when
//! present and falls back to decoding the directory name.
//!
//! ## Environment
//!
//! - `GROK_HOME` — override base directory (default `~/.grok`)
//!
//! ## Resume
//!
//! ```bash
//! grok --resume <session-id>
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::io::BufRead;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde_json::{Value, json};
use tracing::{debug, info, trace, warn};

use crate::discovery::DetectionResult;
use crate::model::{
    CanonicalMessage, CanonicalSession, MessageRole, ToolCall, ToolResult, flatten_content,
    normalize_role, parse_timestamp, reindex_messages, truncate_title,
};
use crate::providers::{Provider, WriteOptions, WrittenSession};

// Thread-local override for GROK_HOME (tests — avoids unsafe env mutation on
// Rust 2024 nightly where the crate forbids `unsafe`).
thread_local! {
    static TEST_GROK_HOME: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

/// Grok Build provider.
pub struct Grok;

impl Grok {
    /// Resolve Grok home: test override → `GROK_HOME` → `~/.grok`.
    fn home_dir() -> Option<PathBuf> {
        let test_override = TEST_GROK_HOME.with(|cell| cell.borrow().clone());
        if let Some(home) = test_override {
            return Some(home);
        }
        Self::home_dir_impl(std::env::var("GROK_HOME").ok())
    }

    fn home_dir_impl(grok_home_env: Option<String>) -> Option<PathBuf> {
        if let Some(home) = grok_home_env {
            let trimmed = home.trim();
            if !trimmed.is_empty() {
                return Some(PathBuf::from(trimmed));
            }
        }
        dirs::home_dir().map(|h| h.join(".grok"))
    }

    fn sessions_dir() -> Option<PathBuf> {
        Self::home_dir().map(|h| h.join("sessions"))
    }

    /// URL-encode a workspace path the same way Grok names session group dirs.
    fn encode_cwd(cwd: &str) -> String {
        urlencoding::encode(cwd).into_owned()
    }

    /// Decode a group directory name back to a cwd path.
    ///
    /// Prefer a `.cwd` sidecar (used when the encoded name was truncated).
    fn decode_cwd_dir(group_dir: &Path) -> Option<PathBuf> {
        let cwd_file = group_dir.join(".cwd");
        if cwd_file.is_file()
            && let Ok(text) = std::fs::read_to_string(&cwd_file)
        {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                return Some(PathBuf::from(trimmed));
            }
        }
        let name = group_dir.file_name()?.to_str()?;
        let decoded = urlencoding::decode(name).ok()?;
        let s = decoded.into_owned();
        if s.starts_with('/') || (s.len() > 2 && s.as_bytes()[1] == b':') {
            Some(PathBuf::from(s))
        } else {
            None
        }
    }

    /// True when `path` looks like a Grok session directory (has summary or
    /// conversation logs).
    fn is_session_dir(path: &Path) -> bool {
        path.is_dir()
            && (path.join("summary.json").is_file()
                || path.join("chat_history.jsonl").is_file()
                || path.join("updates.jsonl").is_file())
    }

    /// Normalize any path pointing at a session file/dir to the session dir.
    fn session_dir_from_path(path: &Path) -> PathBuf {
        if path.is_dir() {
            return path.to_path_buf();
        }
        // summary.json / chat_history.jsonl / updates.jsonl live in the session dir.
        if let Some(name) = path.file_name().and_then(|n| n.to_str())
            && matches!(
                name,
                "summary.json"
                    | "chat_history.jsonl"
                    | "updates.jsonl"
                    | "events.jsonl"
                    | "signals.json"
                    | "plan.json"
            )
            && let Some(parent) = path.parent()
        {
            return parent.to_path_buf();
        }
        path.to_path_buf()
    }

    /// Canonical file path we hand back from owns/list (always a real file).
    fn preferred_session_file(session_dir: &Path) -> Option<PathBuf> {
        for name in ["summary.json", "chat_history.jsonl", "updates.jsonl"] {
            let p = session_dir.join(name);
            if p.is_file() {
                return Some(p);
            }
        }
        None
    }

    /// Walk `sessions/` and collect `(session_id, preferred_file)` pairs.
    fn scan_sessions(root: &Path) -> Vec<(String, PathBuf)> {
        let mut out = Vec::new();
        if !root.is_dir() {
            return out;
        }
        // sessions/<group>/<session-id>/
        let Ok(groups) = std::fs::read_dir(root) else {
            return out;
        };
        for group in groups.flatten() {
            let group_path = group.path();
            if !group_path.is_dir() {
                continue;
            }
            // Skip non-session files at the sessions root (e.g. session_search.sqlite).
            let Ok(children) = std::fs::read_dir(&group_path) else {
                continue;
            };
            for child in children.flatten() {
                let session_dir = child.path();
                if !Self::is_session_dir(&session_dir) {
                    continue;
                }
                let Some(id) = session_dir
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(str::to_string)
                else {
                    continue;
                };
                // Skip clearly non-UUID noise; Grok uses UUIDv7 but clients may
                // supply any UUID. Still accept any non-empty dirname that has
                // session artifacts.
                if id.starts_with('.') {
                    continue;
                }
                if let Some(file) = Self::preferred_session_file(&session_dir) {
                    out.push((id, file));
                }
            }
        }
        out
    }

    /// Load and parse `summary.json` if present.
    fn load_summary(session_dir: &Path) -> Option<Value> {
        let path = session_dir.join("summary.json");
        if !path.is_file() {
            return None;
        }
        let text = std::fs::read_to_string(&path).ok()?;
        serde_json::from_str(&text).ok()
    }

    /// Extract workspace from summary or group directory.
    fn workspace_from(session_dir: &Path, summary: Option<&Value>) -> Option<PathBuf> {
        if let Some(s) = summary {
            if let Some(cwd) = s
                .pointer("/info/cwd")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
            {
                return Some(PathBuf::from(cwd));
            }
            if let Some(cwd) = s
                .get("cwd")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
            {
                return Some(PathBuf::from(cwd));
            }
        }
        session_dir.parent().and_then(Self::decode_cwd_dir)
    }

    /// Read messages preferring `chat_history.jsonl`, falling back to
    /// reconstructing from `updates.jsonl`.
    fn read_messages(session_dir: &Path) -> anyhow::Result<Vec<CanonicalMessage>> {
        let chat_path = session_dir.join("chat_history.jsonl");
        if chat_path.is_file() {
            match Self::read_chat_history(&chat_path) {
                Ok(msgs) if !msgs.is_empty() => return Ok(msgs),
                Ok(_) => {
                    debug!(
                        path = %chat_path.display(),
                        "chat_history.jsonl empty — falling back to updates.jsonl"
                    );
                }
                Err(e) => {
                    warn!(
                        path = %chat_path.display(),
                        error = %e,
                        "failed to parse chat_history.jsonl — falling back to updates.jsonl"
                    );
                }
            }
        }

        let updates_path = session_dir.join("updates.jsonl");
        if updates_path.is_file() {
            return Self::read_updates(&updates_path);
        }

        anyhow::bail!(
            "no chat_history.jsonl or updates.jsonl in {}",
            session_dir.display()
        )
    }

    /// Parse Grok `chat_history.jsonl` into canonical messages.
    fn read_chat_history(path: &Path) -> anyhow::Result<Vec<CanonicalMessage>> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("failed to open {}", path.display()))?;
        let reader = std::io::BufReader::new(file);
        let mut messages = Vec::new();

        for (line_no, line_result) in reader.lines().enumerate() {
            let line = match line_result {
                Ok(l) => l,
                Err(e) => {
                    warn!(line = line_no + 1, error = %e, "skipping unreadable chat_history line");
                    continue;
                }
            };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let val: Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(e) => {
                    warn!(
                        line = line_no + 1,
                        error = %e,
                        "skipping malformed chat_history JSON"
                    );
                    continue;
                }
            };
            if let Some(msg) = Self::chat_entry_to_message(&val) {
                messages.push(msg);
            }
        }

        reindex_messages(&mut messages);
        Ok(messages)
    }

    fn chat_entry_to_message(val: &Value) -> Option<CanonicalMessage> {
        let type_str = val
            .get("type")
            .or_else(|| val.get("role"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        match type_str {
            "system" => {
                let content = flatten_content(val.get("content").unwrap_or(&Value::Null));
                if content.trim().is_empty() {
                    return None;
                }
                Some(CanonicalMessage {
                    idx: 0,
                    role: MessageRole::System,
                    content,
                    timestamp: None,
                    author: Some("system".into()),
                    tool_calls: vec![],
                    tool_results: vec![],
                    extra: val.clone(),
                })
            }
            "user" => {
                let content = flatten_content(val.get("content").unwrap_or(&Value::Null));
                // Grok injects large <user_info>/<git_status> system context as
                // user messages; keep them so conversion preserves context.
                Some(CanonicalMessage {
                    idx: 0,
                    role: MessageRole::User,
                    content,
                    timestamp: None,
                    author: Some("user".into()),
                    tool_calls: vec![],
                    tool_results: vec![],
                    extra: val.clone(),
                })
            }
            "assistant" => {
                let content = flatten_content(val.get("content").unwrap_or(&Value::Null));
                let mut tool_calls = Vec::new();
                if let Some(arr) = val.get("tool_calls").and_then(|v| v.as_array()) {
                    for tc in arr {
                        let name = tc
                            .get("name")
                            .or_else(|| tc.get("function").and_then(|f| f.get("name")))
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string();
                        let id = tc
                            .get("id")
                            .and_then(|v| v.as_str())
                            .map(ToString::to_string);
                        let arguments = tc
                            .get("arguments")
                            .cloned()
                            .or_else(|| {
                                tc.get("function").and_then(|f| f.get("arguments")).cloned()
                            })
                            .map(|a| {
                                // Grok stores arguments as a JSON string.
                                if let Some(s) = a.as_str() {
                                    serde_json::from_str(s).unwrap_or(Value::String(s.to_string()))
                                } else {
                                    a
                                }
                            })
                            .unwrap_or(Value::Null);
                        tool_calls.push(ToolCall {
                            id,
                            name,
                            arguments,
                        });
                    }
                }
                if content.trim().is_empty() && tool_calls.is_empty() {
                    return None;
                }
                let author = val
                    .get("model_id")
                    .and_then(|v| v.as_str())
                    .map(ToString::to_string);
                Some(CanonicalMessage {
                    idx: 0,
                    role: MessageRole::Assistant,
                    content,
                    timestamp: None,
                    author,
                    tool_calls,
                    tool_results: vec![],
                    extra: val.clone(),
                })
            }
            "tool_result" | "tool" | "function" => {
                let call_id = val
                    .get("tool_call_id")
                    .or_else(|| val.get("call_id"))
                    .or_else(|| val.get("id"))
                    .and_then(|v| v.as_str())
                    .map(ToString::to_string);
                let content = flatten_content(val.get("content").unwrap_or(&Value::Null));
                let is_error = val
                    .get("is_error")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                Some(CanonicalMessage {
                    idx: 0,
                    role: MessageRole::Tool,
                    content: content.clone(),
                    timestamp: None,
                    author: Some("tool".into()),
                    tool_calls: vec![],
                    tool_results: vec![ToolResult {
                        call_id,
                        content,
                        is_error,
                    }],
                    extra: val.clone(),
                })
            }
            "reasoning" => {
                // Prefer human-readable summary; fall back to status note.
                let mut parts = Vec::new();
                if let Some(arr) = val.get("summary").and_then(|v| v.as_array()) {
                    for item in arr {
                        if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
                            parts.push(t.to_string());
                        } else if let Some(t) = item.as_str() {
                            parts.push(t.to_string());
                        }
                    }
                }
                if parts.is_empty()
                    && let Some(t) = val.get("content").and_then(|v| v.as_str())
                {
                    parts.push(t.to_string());
                }
                let content = parts.join("\n");
                if content.trim().is_empty() {
                    return None;
                }
                Some(CanonicalMessage {
                    idx: 0,
                    role: MessageRole::Other("reasoning".into()),
                    content,
                    timestamp: None,
                    author: Some("reasoning".into()),
                    tool_calls: vec![],
                    tool_results: vec![],
                    extra: val.clone(),
                })
            }
            other if !other.is_empty() => {
                // Unknown type: try generic role/content extraction.
                let role = normalize_role(other);
                let content = flatten_content(val.get("content").unwrap_or(&Value::Null));
                if content.trim().is_empty() {
                    return None;
                }
                Some(CanonicalMessage {
                    idx: 0,
                    role,
                    content,
                    timestamp: None,
                    author: None,
                    tool_calls: vec![],
                    tool_results: vec![],
                    extra: val.clone(),
                })
            }
            _ => None,
        }
    }

    /// Reconstruct conversation from ACP `updates.jsonl` stream by coalescing
    /// chunks and pairing tool calls with their completed updates.
    fn read_updates(path: &Path) -> anyhow::Result<Vec<CanonicalMessage>> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("failed to open {}", path.display()))?;
        let reader = std::io::BufReader::new(file);

        let mut messages: Vec<CanonicalMessage> = Vec::new();
        let mut user_buf = String::new();
        let mut user_ts: Option<i64> = None;
        let mut agent_buf = String::new();
        let mut agent_ts: Option<i64> = None;
        let mut thought_buf = String::new();
        // toolCallId → (name, arguments, title)
        let mut pending_tools: HashMap<String, (String, Value, Option<String>)> = HashMap::new();

        let flush_user =
            |buf: &mut String, ts: &mut Option<i64>, messages: &mut Vec<CanonicalMessage>| {
                if buf.trim().is_empty() {
                    buf.clear();
                    *ts = None;
                    return;
                }
                messages.push(CanonicalMessage {
                idx: 0,
                role: MessageRole::User,
                content: std::mem::take(buf),
                timestamp: ts.take(),
                author: Some("user".into()),
                tool_calls: vec![],
                tool_results: vec![],
                extra: json!({ "source": "updates.jsonl", "sessionUpdate": "user_message_chunk" }),
            });
            };

        let flush_agent = |buf: &mut String,
                           ts: &mut Option<i64>,
                           messages: &mut Vec<CanonicalMessage>,
                           thought: &mut String| {
            if buf.trim().is_empty() && thought.trim().is_empty() {
                buf.clear();
                thought.clear();
                *ts = None;
                return;
            }
            let content = std::mem::take(buf);
            let mut extra =
                json!({ "source": "updates.jsonl", "sessionUpdate": "agent_message_chunk" });
            if !thought.trim().is_empty() {
                extra["thought"] = Value::String(std::mem::take(thought));
            } else {
                thought.clear();
            }
            messages.push(CanonicalMessage {
                idx: 0,
                role: MessageRole::Assistant,
                content,
                timestamp: ts.take(),
                author: None,
                tool_calls: vec![],
                tool_results: vec![],
                extra,
            });
        };

        for (line_no, line_result) in reader.lines().enumerate() {
            let line = match line_result {
                Ok(l) => l,
                Err(_) => continue,
            };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let val: Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(e) => {
                    warn!(line = line_no + 1, error = %e, "skipping malformed updates line");
                    continue;
                }
            };

            let update = val
                .pointer("/params/update")
                .cloned()
                .or_else(|| val.get("update").cloned())
                .unwrap_or(Value::Null);
            let su = update
                .get("sessionUpdate")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let ts = val
                .pointer("/params/_meta/agentTimestampMs")
                .and_then(parse_timestamp)
                .or_else(|| {
                    val.get("timestamp").and_then(parse_timestamp).map(|t| {
                        // Grok wall-clock timestamps in updates are epoch seconds.
                        if t < 10_000_000_000 { t * 1000 } else { t }
                    })
                });

            match su {
                "user_message_chunk" => {
                    flush_agent(
                        &mut agent_buf,
                        &mut agent_ts,
                        &mut messages,
                        &mut thought_buf,
                    );
                    let text = update
                        .pointer("/content/text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    user_buf.push_str(text);
                    if user_ts.is_none() {
                        user_ts = ts;
                    }
                }
                "agent_message_chunk" => {
                    flush_user(&mut user_buf, &mut user_ts, &mut messages);
                    let text = update
                        .pointer("/content/text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    agent_buf.push_str(text);
                    if agent_ts.is_none() {
                        agent_ts = ts;
                    }
                }
                "agent_thought_chunk" => {
                    flush_user(&mut user_buf, &mut user_ts, &mut messages);
                    let text = update
                        .pointer("/content/text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    thought_buf.push_str(text);
                }
                "tool_call" => {
                    flush_user(&mut user_buf, &mut user_ts, &mut messages);
                    flush_agent(
                        &mut agent_buf,
                        &mut agent_ts,
                        &mut messages,
                        &mut thought_buf,
                    );
                    let id = update
                        .get("toolCallId")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if id.is_empty() {
                        continue;
                    }
                    let name = update
                        .pointer("/_meta/x.ai/tool/name")
                        .or_else(|| update.get("title"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("tool")
                        .to_string();
                    let args = update.get("rawInput").cloned().unwrap_or(Value::Null);
                    let title = update
                        .get("title")
                        .and_then(|v| v.as_str())
                        .map(ToString::to_string);
                    pending_tools.insert(id.clone(), (name.clone(), args.clone(), title));

                    messages.push(CanonicalMessage {
                        idx: 0,
                        role: MessageRole::Assistant,
                        content: String::new(),
                        timestamp: ts,
                        author: None,
                        tool_calls: vec![ToolCall {
                            id: Some(id),
                            name,
                            arguments: args,
                        }],
                        tool_results: vec![],
                        extra: update.clone(),
                    });
                }
                "tool_call_update" => {
                    let id = update
                        .get("toolCallId")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let status = update.get("status").and_then(|v| v.as_str()).unwrap_or("");
                    if status != "completed" && status != "failed" {
                        // Intermediate progress updates — ignore for IR.
                        continue;
                    }
                    flush_user(&mut user_buf, &mut user_ts, &mut messages);
                    flush_agent(
                        &mut agent_buf,
                        &mut agent_ts,
                        &mut messages,
                        &mut thought_buf,
                    );

                    let content = extract_tool_output(&update);
                    let is_error = status == "failed";
                    let (name, _args, _title) = pending_tools
                        .remove(&id)
                        .unwrap_or_else(|| ("tool".into(), Value::Null, None));

                    messages.push(CanonicalMessage {
                        idx: 0,
                        role: MessageRole::Tool,
                        content: content.clone(),
                        timestamp: ts,
                        author: Some(name),
                        tool_calls: vec![],
                        tool_results: vec![ToolResult {
                            call_id: if id.is_empty() { None } else { Some(id) },
                            content,
                            is_error,
                        }],
                        extra: update.clone(),
                    });
                }
                "turn_completed" => {
                    flush_user(&mut user_buf, &mut user_ts, &mut messages);
                    flush_agent(
                        &mut agent_buf,
                        &mut agent_ts,
                        &mut messages,
                        &mut thought_buf,
                    );
                }
                _ => {
                    // plan, hook_execution, subagent_*, compact, etc. — metadata only.
                }
            }
        }

        flush_user(&mut user_buf, &mut user_ts, &mut messages);
        flush_agent(
            &mut agent_buf,
            &mut agent_ts,
            &mut messages,
            &mut thought_buf,
        );
        reindex_messages(&mut messages);
        Ok(messages)
    }

    /// Build chat_history.jsonl lines from canonical messages.
    fn build_chat_history(session: &CanonicalSession) -> String {
        let mut lines = Vec::with_capacity(session.messages.len());
        for msg in &session.messages {
            let line = match &msg.role {
                MessageRole::System => json!({
                    "type": "system",
                    "content": msg.content,
                }),
                MessageRole::User => json!({
                    "type": "user",
                    "content": [{ "type": "text", "text": msg.content }],
                }),
                MessageRole::Assistant => {
                    let mut obj = json!({
                        "type": "assistant",
                        "content": msg.content,
                    });
                    if !msg.tool_calls.is_empty() {
                        let tcs: Vec<Value> = msg
                            .tool_calls
                            .iter()
                            .map(|tc| {
                                let args_str = match &tc.arguments {
                                    Value::String(s) => s.clone(),
                                    other => other.to_string(),
                                };
                                json!({
                                    "id": tc.id.clone().unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                                    "name": tc.name,
                                    "arguments": args_str,
                                })
                            })
                            .collect();
                        obj["tool_calls"] = Value::Array(tcs);
                    }
                    if let Some(model) = session.model_name.as_ref() {
                        obj["model_id"] = Value::String(model.clone());
                    }
                    obj
                }
                MessageRole::Tool => {
                    let call_id = msg
                        .tool_results
                        .first()
                        .and_then(|r| r.call_id.clone())
                        .unwrap_or_default();
                    json!({
                        "type": "tool_result",
                        "tool_call_id": call_id,
                        "content": msg.content,
                    })
                }
                MessageRole::Other(kind) if kind == "reasoning" => json!({
                    "type": "reasoning",
                    "id": format!("rs-casr-{}", msg.idx),
                    "summary": [{ "type": "summary_text", "text": msg.content }],
                    "status": "completed",
                }),
                MessageRole::Other(_) => json!({
                    "type": "user",
                    "content": [{ "type": "text", "text": msg.content }],
                }),
            };
            if let Ok(s) = serde_json::to_string(&line) {
                lines.push(s);
            }
        }
        if lines.is_empty() {
            String::new()
        } else {
            lines.join("\n") + "\n"
        }
    }

    /// Build a minimal but resume-friendly `updates.jsonl` ACP stream.
    fn build_updates(session: &CanonicalSession, session_id: &str) -> String {
        let mut lines = Vec::new();
        let mut event_n: u64 = 1;

        let mut push_update = |update: Value, ts_secs: i64, event_n: &mut u64| {
            let event_id = format!("{session_id}-{}", *event_n);
            *event_n += 1;
            let envelope = json!({
                "timestamp": ts_secs,
                "method": "session/update",
                "params": {
                    "sessionId": session_id,
                    "update": update,
                    "_meta": {
                        "eventId": event_id,
                        "agentTimestampMs": ts_secs * 1000,
                    }
                }
            });
            if let Ok(s) = serde_json::to_string(&envelope) {
                lines.push(s);
            }
        };

        for msg in &session.messages {
            let ts_secs = msg
                .timestamp
                .map(|ms| if ms > 10_000_000_000 { ms / 1000 } else { ms })
                .or_else(|| session.started_at.map(|ms| ms / 1000))
                .unwrap_or_else(|| chrono::Utc::now().timestamp());

            match &msg.role {
                MessageRole::User => {
                    push_update(
                        json!({
                            "sessionUpdate": "user_message_chunk",
                            "content": { "type": "text", "text": msg.content },
                        }),
                        ts_secs,
                        &mut event_n,
                    );
                }
                MessageRole::Assistant => {
                    if !msg.content.is_empty() {
                        push_update(
                            json!({
                                "sessionUpdate": "agent_message_chunk",
                                "content": { "type": "text", "text": msg.content },
                            }),
                            ts_secs,
                            &mut event_n,
                        );
                    }
                    for tc in &msg.tool_calls {
                        let id = tc
                            .id
                            .clone()
                            .unwrap_or_else(|| format!("call-casr-{}", uuid::Uuid::new_v4()));
                        push_update(
                            json!({
                                "sessionUpdate": "tool_call",
                                "toolCallId": id,
                                "title": tc.name,
                                "rawInput": tc.arguments,
                                "_meta": {
                                    "x.ai/tool": {
                                        "version": 1,
                                        "name": tc.name,
                                        "kind": "other",
                                        "namespace": "casr",
                                        "label": tc.name,
                                        "read_only": true
                                    }
                                }
                            }),
                            ts_secs,
                            &mut event_n,
                        );
                    }
                }
                MessageRole::Tool => {
                    for tr in &msg.tool_results {
                        let id = tr
                            .call_id
                            .clone()
                            .unwrap_or_else(|| format!("call-casr-{}", uuid::Uuid::new_v4()));
                        let status = if tr.is_error { "failed" } else { "completed" };
                        push_update(
                            json!({
                                "sessionUpdate": "tool_call_update",
                                "toolCallId": id,
                                "status": status,
                                "content": [{
                                    "type": "content",
                                    "content": { "type": "text", "text": tr.content }
                                }],
                                "rawOutput": { "type": "Text", "Content": tr.content },
                            }),
                            ts_secs,
                            &mut event_n,
                        );
                    }
                    // If tool_results empty, still emit content as a completed update.
                    if msg.tool_results.is_empty() && !msg.content.is_empty() {
                        push_update(
                            json!({
                                "sessionUpdate": "tool_call_update",
                                "toolCallId": format!("call-casr-{}", msg.idx),
                                "status": "completed",
                                "content": [{
                                    "type": "content",
                                    "content": { "type": "text", "text": msg.content }
                                }],
                            }),
                            ts_secs,
                            &mut event_n,
                        );
                    }
                }
                MessageRole::System | MessageRole::Other(_) => {
                    // System/reasoning are not part of the ACP user-facing stream.
                }
            }
        }

        // Close with turn_completed so Grok knows the stream is idle.
        let ts_secs = session
            .ended_at
            .or(session.started_at)
            .map(|ms| ms / 1000)
            .unwrap_or_else(|| chrono::Utc::now().timestamp());
        push_update(
            json!({
                "sessionUpdate": "turn_completed",
                "prompt_id": uuid::Uuid::new_v4().to_string(),
                "stop_reason": "end_turn"
            }),
            ts_secs,
            &mut event_n,
        );

        if lines.is_empty() {
            String::new()
        } else {
            lines.join("\n") + "\n"
        }
    }

    /// Build `summary.json` metadata.
    fn build_summary(
        session: &CanonicalSession,
        session_id: &str,
        cwd: &str,
        chat_msg_count: usize,
        update_event_count: usize,
    ) -> Value {
        let now = chrono::Utc::now();
        let rfc3339 = |ms: Option<i64>| {
            ms.and_then(chrono::DateTime::from_timestamp_millis)
                .unwrap_or(now)
                .to_rfc3339_opts(chrono::SecondsFormat::Micros, true)
        };
        let title = session
            .title
            .clone()
            .or_else(|| {
                session
                    .messages
                    .iter()
                    .find(|m| m.role == MessageRole::User)
                    .map(|m| truncate_title(&m.content, 80))
            })
            .unwrap_or_else(|| format!("casr import {session_id}"));

        let model = session
            .model_name
            .clone()
            .unwrap_or_else(|| "grok-4.5".into());

        let grok_home = Self::home_dir()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| "~/.grok".into());

        json!({
            "info": {
                "id": session_id,
                "cwd": cwd,
            },
            "session_summary": title,
            "generated_title": title,
            "created_at": rfc3339(session.started_at),
            "updated_at": rfc3339(session.ended_at),
            "last_active_at": rfc3339(session.ended_at),
            "num_messages": update_event_count,
            "num_chat_messages": chat_msg_count,
            "current_model_id": model,
            "next_trace_turn": 1,
            "chat_format_version": 1,
            "grok_home": grok_home,
            "agent_name": "casr-import",
            "sandbox_profile": "off",
            "reasoning_effort": "high",
            "casr": {
                "source_provider": session.provider_slug,
                "source_session_id": session.session_id,
                "imported_at": now.to_rfc3339_opts(chrono::SecondsFormat::Micros, true),
            }
        })
    }
}

/// Pull human-readable tool output from a `tool_call_update` payload.
fn extract_tool_output(update: &Value) -> String {
    // Prefer rawOutput string forms.
    if let Some(raw) = update.get("rawOutput") {
        if let Some(s) = raw.as_str() {
            return s.to_string();
        }
        // { "type": "ListDir", "Content": { "content": "..." } }
        if let Some(s) = raw.pointer("/Content/content").and_then(|v| v.as_str()) {
            return s.to_string();
        }
        if let Some(s) = raw.pointer("/Content").and_then(|v| v.as_str()) {
            return s.to_string();
        }
        if let Some(s) = raw.get("content").and_then(|v| v.as_str()) {
            return s.to_string();
        }
        if let Some(s) = raw.get("output").and_then(|v| v.as_str()) {
            return s.to_string();
        }
        // Last resort: compact JSON.
        if let Ok(s) = serde_json::to_string(raw)
            && s != "null"
            && s != "{}"
        {
            return s;
        }
    }

    // content: [ { type: "content", content: { type: "text", text: "..." } } ]
    if let Some(arr) = update.get("content").and_then(|v| v.as_array()) {
        let mut parts = Vec::new();
        for item in arr {
            if let Some(t) = item.pointer("/content/text").and_then(|v| v.as_str()) {
                parts.push(t.to_string());
            } else if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
                parts.push(t.to_string());
            } else {
                let flat = flatten_content(item);
                if !flat.is_empty() {
                    parts.push(flat);
                }
            }
        }
        if !parts.is_empty() {
            return parts.join("\n");
        }
    }

    String::new()
}

impl Provider for Grok {
    fn name(&self) -> &str {
        "Grok"
    }

    fn slug(&self) -> &str {
        "grok"
    }

    fn cli_alias(&self) -> &str {
        "grk"
    }

    fn detect(&self) -> DetectionResult {
        let mut evidence = Vec::new();
        let mut installed = false;

        if which::which("grok").is_ok() {
            evidence.push("grok binary found in PATH".to_string());
            installed = true;
        }
        // Also detect the default install location used by Grok's installer.
        if let Some(home) = dirs::home_dir() {
            let bundled = home.join(".grok").join("bin").join("grok");
            if bundled.is_file() {
                evidence.push(format!("grok binary found at {}", bundled.display()));
                installed = true;
            }
        }
        if let Some(sessions) = Self::sessions_dir()
            && sessions.is_dir()
        {
            evidence.push(format!("sessions directory found: {}", sessions.display()));
            installed = true;
        }
        if let Some(home) = Self::home_dir()
            && home.is_dir()
        {
            evidence.push(format!("GROK home found: {}", home.display()));
            installed = true;
        }

        trace!(provider = "grok", ?evidence, installed, "detection");
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
        let root = Self::sessions_dir()?;
        Some(Self::scan_sessions(&root))
    }

    fn owns_session(&self, session_id: &str) -> Option<PathBuf> {
        let root = Self::sessions_dir()?;
        if !root.is_dir() {
            return None;
        }
        // Fast path: walk group dirs looking for exact session-id child.
        let Ok(groups) = std::fs::read_dir(&root) else {
            return None;
        };
        for group in groups.flatten() {
            let session_dir = group.path().join(session_id);
            if Self::is_session_dir(&session_dir)
                && let Some(file) = Self::preferred_session_file(&session_dir)
            {
                debug!(
                    provider = "grok",
                    path = %file.display(),
                    session_id,
                    "owns session"
                );
                return Some(file);
            }
        }
        None
    }

    fn read_session(&self, path: &Path) -> anyhow::Result<CanonicalSession> {
        let session_dir = Self::session_dir_from_path(path);
        debug!(path = %session_dir.display(), "reading Grok session");

        if !session_dir.exists() {
            anyhow::bail!(
                "Grok session path does not exist: {}",
                session_dir.display()
            );
        }

        let summary = Self::load_summary(&session_dir);
        let session_id = summary
            .as_ref()
            .and_then(|s| s.pointer("/info/id").and_then(|v| v.as_str()))
            .map(ToString::to_string)
            .or_else(|| {
                session_dir
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(ToString::to_string)
            })
            .unwrap_or_else(|| "unknown".into());

        let workspace = Self::workspace_from(&session_dir, summary.as_ref());
        let title = summary.as_ref().and_then(|s| {
            s.get("generated_title")
                .or_else(|| s.get("session_summary"))
                .or_else(|| s.get("title"))
                .and_then(|v| v.as_str())
                .filter(|t| !t.is_empty())
                .map(ToString::to_string)
        });
        let started_at = summary
            .as_ref()
            .and_then(|s| s.get("created_at").and_then(parse_timestamp));
        let ended_at = summary.as_ref().and_then(|s| {
            s.get("last_active_at")
                .or_else(|| s.get("updated_at"))
                .and_then(parse_timestamp)
        });
        let model_name = summary.as_ref().and_then(|s| {
            s.get("current_model_id")
                .and_then(|v| v.as_str())
                .map(ToString::to_string)
        });

        let mut messages = Self::read_messages(&session_dir)?;
        // Stamp timestamps from summary onto first/last when messages lack them.
        if let Some(start) = started_at
            && let Some(first) = messages.first_mut()
            && first.timestamp.is_none()
        {
            first.timestamp = Some(start);
        }
        if let Some(end) = ended_at
            && let Some(last) = messages.last_mut()
            && last.timestamp.is_none()
        {
            last.timestamp = Some(end);
        }

        let title = title.or_else(|| {
            messages
                .iter()
                .find(|m| m.role == MessageRole::User && !m.content.trim().is_empty())
                .map(|m| truncate_title(&m.content, 100))
        });

        let mut meta = serde_json::Map::new();
        meta.insert("source".into(), json!("grok"));
        if let Some(s) = &summary {
            for key in [
                "agent_name",
                "reasoning_effort",
                "sandbox_profile",
                "chat_format_version",
                "git_root_dir",
                "git_remotes",
                "head_commit",
                "head_branch",
                "session_kind",
            ] {
                if let Some(v) = s.get(key) {
                    meta.insert(key.to_string(), v.clone());
                }
            }
        }

        info!(session_id, messages = messages.len(), "Grok session parsed");

        Ok(CanonicalSession {
            session_id,
            provider_slug: "grok".to_string(),
            workspace,
            title,
            started_at,
            ended_at,
            messages,
            metadata: Value::Object(meta),
            source_path: path.to_path_buf(),
            model_name,
        })
    }

    fn write_session(
        &self,
        session: &CanonicalSession,
        opts: &WriteOptions,
    ) -> anyhow::Result<WrittenSession> {
        let sessions_root = Self::sessions_dir()
            .ok_or_else(|| anyhow::anyhow!("cannot determine Grok home directory"))?;

        let session_id = opts
            .target_session_id
            .clone()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let cwd = session
            .workspace
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned())
            .or_else(|| {
                std::env::current_dir()
                    .ok()
                    .map(|p| p.to_string_lossy().into_owned())
            })
            .unwrap_or_else(|| "/".into());

        let group = Self::encode_cwd(&cwd);
        let session_dir = sessions_root.join(&group).join(&session_id);

        // Conflict check on the session directory itself.
        if session_dir.exists() && !opts.force {
            return Err(crate::error::CasrError::SessionConflict {
                session_id: session_id.clone(),
                existing_path: session_dir,
            }
            .into());
        }

        std::fs::create_dir_all(&session_dir).with_context(|| {
            format!(
                "failed to create Grok session directory: {}",
                session_dir.display()
            )
        })?;

        let chat = Self::build_chat_history(session);
        let updates = Self::build_updates(session, &session_id);
        let chat_count = chat.lines().filter(|l| !l.trim().is_empty()).count();
        let update_count = updates.lines().filter(|l| !l.trim().is_empty()).count();
        let summary = Self::build_summary(session, &session_id, &cwd, chat_count, update_count);
        let summary_bytes = serde_json::to_vec_pretty(&summary)?;

        let summary_path = session_dir.join("summary.json");
        let chat_path = session_dir.join("chat_history.jsonl");
        let updates_path = session_dir.join("updates.jsonl");

        let summary_out =
            crate::pipeline::atomic_write(&summary_path, &summary_bytes, opts.force, self.slug())?;
        let chat_out =
            crate::pipeline::atomic_write(&chat_path, chat.as_bytes(), true, self.slug())?;
        let updates_out =
            crate::pipeline::atomic_write(&updates_path, updates.as_bytes(), true, self.slug())?;

        // Best-effort empty signals so tools that read them don't panic.
        let signals = json!({
            "turnCount": session.messages.iter().filter(|m| m.role == MessageRole::User).count(),
            "userMessageCount": session.messages.iter().filter(|m| m.role == MessageRole::User).count(),
            "assistantMessageCount": session.messages.iter().filter(|m| m.role == MessageRole::Assistant).count(),
            "toolCallCount": session.messages.iter().map(|m| m.tool_calls.len()).sum::<usize>(),
            "primaryModelId": session.model_name.clone().unwrap_or_else(|| "grok-4.5".into()),
            "modelsUsed": session.model_name.as_ref().map(|m| vec![m.clone()]).unwrap_or_default(),
            "importedBy": "casr",
        });
        if let Ok(bytes) = serde_json::to_vec_pretty(&signals) {
            let _ = crate::pipeline::atomic_write(
                &session_dir.join("signals.json"),
                &bytes,
                true,
                self.slug(),
            );
        }

        info!(
            session_id,
            path = %session_dir.display(),
            messages = session.messages.len(),
            "Grok session written"
        );

        Ok(WrittenSession {
            paths: vec![
                summary_out.target_path,
                chat_out.target_path,
                updates_out.target_path,
            ],
            session_id: session_id.clone(),
            resume_command: self.resume_command(&session_id),
            backup_path: summary_out.backup_path,
            warnings: Vec::new(),
        })
    }

    fn resume_command(&self, session_id: &str) -> String {
        format!("grok --resume {session_id}")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{CanonicalMessage, CanonicalSession, MessageRole, ToolCall, ToolResult};
    use crate::providers::Provider;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn with_temp_home<F, R>(f: F) -> R
    where
        F: FnOnce(&Path) -> R,
    {
        let n = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("casr-grok-test-{}-{}", std::process::id(), n));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sessions")).unwrap();
        TEST_GROK_HOME.with(|cell| *cell.borrow_mut() = Some(dir.clone()));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&dir)));
        TEST_GROK_HOME.with(|cell| *cell.borrow_mut() = None);
        let _ = std::fs::remove_dir_all(&dir);
        match result {
            Ok(r) => r,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    fn write_fixture_session(
        home: &Path,
        cwd: &str,
        id: &str,
        summary: &str,
        chat: &str,
        updates: &str,
    ) {
        let dir = home.join("sessions").join(Grok::encode_cwd(cwd)).join(id);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("summary.json"), summary).unwrap();
        std::fs::write(dir.join("chat_history.jsonl"), chat).unwrap();
        std::fs::write(dir.join("updates.jsonl"), updates).unwrap();
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
            extra: Value::Null,
        }
    }

    #[test]
    fn encode_cwd_percent_encodes_slashes() {
        let enc = Grok::encode_cwd("/Users/me/proj");
        assert!(
            enc.contains("%2F"),
            "expected percent-encoded path, got {enc}"
        );
        assert!(!enc.contains('/'));
    }

    #[test]
    fn resume_command_shape() {
        assert_eq!(
            Grok.resume_command("019f4f55-3ff1-7f52-8606-fbc86c04ead3"),
            "grok --resume 019f4f55-3ff1-7f52-8606-fbc86c04ead3"
        );
    }

    #[test]
    fn detect_and_list_and_owns() {
        with_temp_home(|home| {
            write_fixture_session(
                home,
                "/tmp/proj",
                "sess-aaa",
                r#"{"info":{"id":"sess-aaa","cwd":"/tmp/proj"},"created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:01:00Z","session_summary":"hi","generated_title":"hi","current_model_id":"grok-4.5","num_messages":2,"num_chat_messages":2,"chat_format_version":1}"#,
                r#"{"type":"user","content":[{"type":"text","text":"hello"}]}
{"type":"assistant","content":"world","model_id":"grok-4.5"}
"#,
                r#"{"timestamp":1700000000,"method":"session/update","params":{"sessionId":"sess-aaa","update":{"sessionUpdate":"user_message_chunk","content":{"type":"text","text":"hello"}}}}
{"timestamp":1700000001,"method":"session/update","params":{"sessionId":"sess-aaa","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"world"}}}}
"#,
            );

            let det = Grok.detect();
            assert!(det.installed, "expected installed with sessions dir");

            let listed = Grok.list_sessions().expect("list_sessions");
            assert_eq!(listed.len(), 1);
            assert_eq!(listed[0].0, "sess-aaa");

            let owned = Grok.owns_session("sess-aaa");
            assert!(owned.is_some());
            assert!(owned.unwrap().ends_with("summary.json"));
            assert!(Grok.owns_session("nope").is_none());
        });
    }

    #[test]
    fn read_chat_history_with_tools() {
        with_temp_home(|home| {
            let chat = r#"{"type":"user","content":[{"type":"text","text":"list files"}]}
{"type":"assistant","content":"Sure","tool_calls":[{"id":"call-1","name":"list_dir","arguments":"{\"target_directory\":\"/tmp\"}"}],"model_id":"grok-4.5"}
{"type":"tool_result","tool_call_id":"call-1","content":"a\nb\n"}
{"type":"assistant","content":"done"}
"#;
            write_fixture_session(
                home,
                "/tmp/proj",
                "sess-tools",
                r#"{"info":{"id":"sess-tools","cwd":"/tmp/proj"},"created_at":"2026-01-01T00:00:00.000000Z","updated_at":"2026-01-01T00:05:00.000000Z","generated_title":"List Files","session_summary":"List Files","current_model_id":"grok-4.5","num_messages":4,"num_chat_messages":4,"chat_format_version":1}"#,
                chat,
                "",
            );
            let path = Grok.owns_session("sess-tools").unwrap();
            let session = Grok.read_session(&path).unwrap();
            assert_eq!(session.session_id, "sess-tools");
            assert_eq!(session.model_name.as_deref(), Some("grok-4.5"));
            assert_eq!(session.workspace, Some(PathBuf::from("/tmp/proj")));
            assert_eq!(session.title.as_deref(), Some("List Files"));
            // user, assistant(+tool), tool_result, assistant
            assert_eq!(session.messages.len(), 4);
            assert_eq!(session.messages[0].role, MessageRole::User);
            assert_eq!(session.messages[1].role, MessageRole::Assistant);
            assert_eq!(session.messages[1].tool_calls.len(), 1);
            assert_eq!(session.messages[1].tool_calls[0].name, "list_dir");
            assert_eq!(session.messages[2].role, MessageRole::Tool);
            assert_eq!(
                session.messages[2].tool_results[0].call_id.as_deref(),
                Some("call-1")
            );
            assert_eq!(session.messages[3].content, "done");
        });
    }

    #[test]
    fn read_updates_fallback_coalesces_chunks() {
        with_temp_home(|home| {
            // No chat_history — force updates path.
            let dir = home
                .join("sessions")
                .join(Grok::encode_cwd("/tmp/u"))
                .join("sess-upd");
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                dir.join("summary.json"),
                r#"{"info":{"id":"sess-upd","cwd":"/tmp/u"},"created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:10Z","generated_title":"Upd","current_model_id":"grok-4.5","chat_format_version":1}"#,
            )
            .unwrap();
            let updates = r#"{"timestamp":1700000000,"method":"session/update","params":{"sessionId":"sess-upd","update":{"sessionUpdate":"user_message_chunk","content":{"type":"text","text":"hel"}},"_meta":{"agentTimestampMs":1700000000000}}}
{"timestamp":1700000000,"method":"session/update","params":{"sessionId":"sess-upd","update":{"sessionUpdate":"user_message_chunk","content":{"type":"text","text":"lo"}}}}
{"timestamp":1700000001,"method":"session/update","params":{"sessionId":"sess-upd","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"hi "}}}}
{"timestamp":1700000001,"method":"session/update","params":{"sessionId":"sess-upd","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"there"}}}}
{"timestamp":1700000002,"method":"session/update","params":{"sessionId":"sess-upd","update":{"sessionUpdate":"tool_call","toolCallId":"call-x","title":"list_dir","rawInput":{"target_directory":"/tmp"},"_meta":{"x.ai/tool":{"name":"list_dir"}}}}}
{"timestamp":1700000003,"method":"session/update","params":{"sessionId":"sess-upd","update":{"sessionUpdate":"tool_call_update","toolCallId":"call-x","status":"completed","rawOutput":{"type":"ListDir","Content":{"content":"file1"}}}}}
{"timestamp":1700000004,"method":"_x.ai/session/update","params":{"sessionId":"sess-upd","update":{"sessionUpdate":"turn_completed","stop_reason":"end_turn"}}}
"#;
            std::fs::write(dir.join("updates.jsonl"), updates).unwrap();

            let path = dir.join("summary.json");
            let session = Grok.read_session(&path).unwrap();
            // user (coalesced) + assistant text + assistant tool_call + tool result
            assert_eq!(
                session.messages.len(),
                4,
                "got messages: {:?}",
                session
                    .messages
                    .iter()
                    .map(|m| format!(
                        "{:?} content={:?} tools={}",
                        m.role,
                        m.content,
                        m.tool_calls.len()
                    ))
                    .collect::<Vec<_>>()
            );
            assert_eq!(session.messages[0].role, MessageRole::User);
            assert_eq!(session.messages[0].content, "hello");
            assert_eq!(session.messages[1].role, MessageRole::Assistant);
            assert_eq!(session.messages[1].content, "hi there");
            assert_eq!(session.messages[2].role, MessageRole::Assistant);
            assert_eq!(session.messages[2].tool_calls.len(), 1);
            assert_eq!(session.messages[2].tool_calls[0].name, "list_dir");
            assert_eq!(session.messages[3].role, MessageRole::Tool);
            assert!(session.messages[3].content.contains("file1"));
        });
    }

    #[test]
    fn write_then_read_round_trip() {
        with_temp_home(|_home| {
            let mut messages = vec![
                msg(MessageRole::User, "what is 2+2?"),
                msg(MessageRole::Assistant, "4"),
            ];
            messages[1].tool_calls = vec![ToolCall {
                id: Some("call-math".into()),
                name: "calculator".into(),
                arguments: json!({"expr": "2+2"}),
            }];
            messages.push(CanonicalMessage {
                idx: 0,
                role: MessageRole::Tool,
                content: "4".into(),
                timestamp: Some(1_700_000_001_000),
                author: None,
                tool_calls: vec![],
                tool_results: vec![ToolResult {
                    call_id: Some("call-math".into()),
                    content: "4".into(),
                    is_error: false,
                }],
                extra: Value::Null,
            });
            messages.push(msg(MessageRole::Assistant, "The answer is 4."));

            let canonical = CanonicalSession {
                session_id: "src-1".into(),
                provider_slug: "claude-code".into(),
                workspace: Some(PathBuf::from("/tmp/roundtrip")),
                title: Some("Math".into()),
                started_at: Some(1_700_000_000_000),
                ended_at: Some(1_700_000_010_000),
                messages,
                metadata: json!({}),
                source_path: PathBuf::from("/tmp/src"),
                model_name: Some("grok-4.5".into()),
            };

            let written = Grok
                .write_session(
                    &canonical,
                    &WriteOptions {
                        force: false,
                        target_session_id: Some("written-sess-1".into()),
                    },
                )
                .unwrap();
            assert_eq!(written.session_id, "written-sess-1");
            assert_eq!(written.resume_command, "grok --resume written-sess-1");
            assert!(written.paths.len() >= 3);

            let path = Grok.owns_session("written-sess-1").unwrap();
            let back = Grok.read_session(&path).unwrap();
            assert_eq!(back.session_id, "written-sess-1");
            assert_eq!(back.workspace, Some(PathBuf::from("/tmp/roundtrip")));
            assert_eq!(back.title.as_deref(), Some("Math"));
            // system not written; user + assistant(+tools) + tool + assistant
            let roles: Vec<_> = back.messages.iter().map(|m| &m.role).collect();
            assert!(
                roles.iter().any(|r| matches!(r, MessageRole::User)),
                "missing user: {roles:?}"
            );
            assert!(
                roles.iter().any(|r| matches!(r, MessageRole::Assistant)),
                "missing assistant: {roles:?}"
            );
            assert!(
                roles.iter().any(|r| matches!(r, MessageRole::Tool)),
                "missing tool: {roles:?}"
            );
            let user = back
                .messages
                .iter()
                .find(|m| m.role == MessageRole::User)
                .unwrap();
            assert_eq!(user.content, "what is 2+2?");
        });
    }

    #[test]
    fn write_conflict_without_force() {
        with_temp_home(|_home| {
            let session = CanonicalSession {
                session_id: "x".into(),
                provider_slug: "test".into(),
                workspace: Some(PathBuf::from("/tmp/c")),
                title: None,
                started_at: None,
                ended_at: None,
                messages: vec![
                    msg(MessageRole::User, "a"),
                    msg(MessageRole::Assistant, "b"),
                ],
                metadata: json!({}),
                source_path: PathBuf::from("/tmp"),
                model_name: None,
            };
            let opts = WriteOptions {
                force: false,
                target_session_id: Some("conflict-id".into()),
            };
            Grok.write_session(&session, &opts).unwrap();
            let err = Grok.write_session(&session, &opts).unwrap_err();
            let msg = format!("{err:#}");
            assert!(
                msg.to_lowercase().contains("conflict") || msg.contains("already"),
                "unexpected error: {msg}"
            );
            // force should succeed
            let ok = Grok
                .write_session(
                    &session,
                    &WriteOptions {
                        force: true,
                        target_session_id: Some("conflict-id".into()),
                    },
                )
                .unwrap();
            assert_eq!(ok.session_id, "conflict-id");
        });
    }

    #[test]
    fn extract_tool_output_variants() {
        let list_dir = json!({
            "rawOutput": { "type": "ListDir", "Content": { "content": "fileA" } }
        });
        assert_eq!(extract_tool_output(&list_dir), "fileA");

        let text = json!({
            "content": [{ "type": "content", "content": { "type": "text", "text": "out" } }]
        });
        assert_eq!(extract_tool_output(&text), "out");
    }

    #[test]
    fn home_dir_impl_prefers_env() {
        assert_eq!(
            Grok::home_dir_impl(Some("/custom/grok".into())),
            Some(PathBuf::from("/custom/grok"))
        );
        assert!(Grok::home_dir_impl(None).is_some());
    }
}

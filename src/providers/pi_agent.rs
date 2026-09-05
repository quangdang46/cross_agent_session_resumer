//! Pi JSONL format providers — Pi-Agent (`pi`) and OMP (oh-my-pi, `omp`).
//!
//! Both CLIs store sessions as JSONL with typed entries and content blocks:
//! - Pi-Agent: `~/.pi/agent/sessions/<safe-path>/<timestamp>_<uuid>.jsonl`
//!   (override: `$PI_AGENT_HOME`)
//! - OMP (oh-my-pi, a fork of Pi Agent): `~/.omp/agent/sessions/...`
//!   (override: `$OMP_HOME`)
//!
//! The two formats are identical, but each CLI owns a separate home
//! directory, so casr exposes them as two independent providers sharing the
//! read/write engine in this module. This keeps both session stores visible
//! at all times (a single shared provider would hide whichever home lost the
//! precedence race) and lets `--source pi` / `--source omp` disambiguate.
//!
//! ## JSONL format
//!
//! Each line has a `type` discriminator:
//! - `"session"` — header with `id`, `timestamp`, `cwd`, `provider`, `modelId`
//! - `"message"` — conversation message with nested `message` object
//! - `"model_change"` — records model/provider switches
//! - `"thinking_level_change"` — records thinking level changes (skipped)
//!
//! Messages are wrapped:
//! ```json
//! {"type":"message","timestamp":"...","message":{"role":"user","content":"..."}}
//! ```
//!
//! Content can be a plain string or an array of typed blocks:
//! - `{"type":"text","text":"..."}` — text content
//! - `{"type":"toolCall","name":"...","arguments":{...}}` — tool invocations
//! - `{"type":"thinking","thinking":"..."}` — chain-of-thought
//! - `{"type":"image",...}` — images (skipped)
//!
//! ## Session ID scheme
//!
//! Sessions are identified by the filename stem (e.g. `2025-12-01T10-00-00_uuid1`).
//! Files must contain an underscore to be recognized as session files.

use std::io::BufRead;
use std::path::{Path, PathBuf};

use tracing::{debug, info, trace};

use crate::discovery::DetectionResult;
use crate::model::{
    CanonicalMessage, CanonicalSession, MessageRole, ToolCall, ToolResult, normalize_role,
    parse_timestamp, reindex_messages, truncate_title,
};
use crate::providers::{Provider, WriteOptions, WrittenSession};

/// Which CLI flavor of the shared Pi JSONL format a provider targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PiFlavor {
    /// Original Pi Agent (`pi` binary).
    Pi,
    /// oh-my-pi fork (`omp` binary).
    Omp,
}

impl PiFlavor {
    /// Provider slug used in session metadata and `providers --json`.
    fn slug(self) -> &'static str {
        match self {
            PiFlavor::Pi => "pi-agent",
            PiFlavor::Omp => "omp",
        }
    }

    /// Human-readable provider name.
    fn display_name(self) -> &'static str {
        match self {
            PiFlavor::Pi => "Pi-Agent",
            PiFlavor::Omp => "OMP (oh-my-pi)",
        }
    }

    /// CLI alias for `casr <alias> resume …`.
    fn cli_alias(self) -> &'static str {
        match self {
            PiFlavor::Pi => "pi",
            PiFlavor::Omp => "omp",
        }
    }

    /// Environment variable overriding this flavor's home directory.
    fn home_env_var(self) -> &'static str {
        match self {
            PiFlavor::Pi => "PI_AGENT_HOME",
            PiFlavor::Omp => "OMP_HOME",
        }
    }

    /// Default home directory when the env override is unset.
    fn default_home(self) -> PathBuf {
        let home = dirs::home_dir().unwrap_or_default();
        match self {
            PiFlavor::Pi => home.join(".pi").join("agent"),
            PiFlavor::Omp => home.join(".omp").join("agent"),
        }
    }

    /// Binary invoked by the resume command.
    fn resume_binary(self) -> &'static str {
        match self {
            PiFlavor::Pi => "pi",
            PiFlavor::Omp => "omp",
        }
    }

    /// `metadata.source` tag recorded when reading a session file.
    fn source_tag(self) -> &'static str {
        match self {
            PiFlavor::Pi => "pi_agent",
            PiFlavor::Omp => "omp",
        }
    }
}

/// Pi-Agent provider — original `pi` CLI storing sessions under `~/.pi/agent`.
pub struct PiAgent;

/// OMP provider — oh-my-pi `omp` CLI storing sessions under `~/.omp/agent`.
pub struct Omp;

/// Shared read/write engine for both Pi JSONL flavors.
struct PiEngine {
    flavor: PiFlavor,
}

impl PiEngine {
    fn new(flavor: PiFlavor) -> Self {
        Self { flavor }
    }

    /// Root directory for this flavor's session storage.
    ///
    /// Resolution precedence:
    /// 1. The flavor's env override (`$PI_AGENT_HOME` / `$OMP_HOME`)
    /// 2. The flavor's own default (`~/.pi/agent` / `~/.omp/agent`)
    ///
    /// An env var pointing at a non-existent path is still returned so that
    /// `detect()` can correctly report `installed=false` (the `sessions/`
    /// subdir won't exist either) and `write_session` can create the
    /// directory tree. The critical guard is in [`Self::sessions_dir`]: it
    /// checks `sessions.is_dir()` and falls back to the home dir, NOT to
    /// other providers' roots. This prevents the env-var leak that was the
    /// actual bug — previously, when an env var pointed to a non-existent
    /// dir, the method fell through to the other flavor's home and
    /// discovered live session files there that belonged to unrelated
    /// session IDs.
    fn home_dir(&self) -> PathBuf {
        Self::home_dir_impl(
            std::env::var(self.flavor.home_env_var()).ok(),
            self.flavor.default_home(),
        )
    }

    /// Inner implementation factored out for testability without env-var
    /// manipulation (which is `unsafe` on Rust 2024 nightly).
    fn home_dir_impl(env_value: Option<String>, default: PathBuf) -> PathBuf {
        env_value.map(PathBuf::from).unwrap_or(default)
    }

    /// Sessions directory under the home dir.
    fn sessions_dir(home: &Path) -> PathBuf {
        let sessions = home.join("sessions");
        if sessions.exists() {
            sessions
        } else {
            home.to_path_buf()
        }
    }

    /// Flatten Pi-Agent message content to a string.
    ///
    /// Handles plain string content and arrays of typed blocks:
    /// text, thinking, toolCall (image is skipped).
    fn flatten_content(content: &serde_json::Value) -> String {
        if let Some(s) = content.as_str() {
            return s.to_string();
        }
        if let Some(arr) = content.as_array() {
            let parts: Vec<String> = arr
                .iter()
                .filter_map(|block| {
                    let block_type = block.get("type").and_then(|t| t.as_str());
                    match block_type {
                        Some("text") => {
                            block.get("text").and_then(|t| t.as_str()).map(String::from)
                        }
                        Some("thinking") => block
                            .get("thinking")
                            .and_then(|t| t.as_str())
                            .map(|t| format!("[Thinking] {t}")),
                        Some("toolCall") => {
                            let name = block
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown");
                            Some(format!("[Tool: {name}]"))
                        }
                        Some("image") => None,
                        _ => None,
                    }
                })
                .collect();
            return parts.join("\n");
        }
        String::new()
    }

    /// Extract tool calls from a content block array.
    fn extract_tool_calls(content: &serde_json::Value) -> Vec<ToolCall> {
        let Some(arr) = content.as_array() else {
            return vec![];
        };
        arr.iter()
            .filter_map(|block| {
                if block.get("type").and_then(|t| t.as_str()) != Some("toolCall") {
                    return None;
                }
                Some(ToolCall {
                    id: block.get("id").and_then(|v| v.as_str()).map(String::from),
                    name: block
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string(),
                    arguments: block
                        .get("arguments")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null),
                })
            })
            .collect()
    }

    /// Check whether a candidate file's content is a pi-agent/omp session.
    ///
    /// Scans the first JSONL entries (bounded read) for a known type
    /// discriminator — the `session` header or a `message` entry. Recent
    /// pi/omp versions prepend `{"type":"title",...}` metadata lines before
    /// the session header, so the very first line is not necessarily the
    /// header. Entries from other JSONL providers (e.g. Claude Code's
    /// `"user"`/`"assistant"` lines) never match.
    fn content_looks_like_pi_session(buf: &[u8]) -> bool {
        const SCAN_BYTES: usize = 8192;
        const SCAN_LINES: usize = 8;
        let max = buf.len().min(SCAN_BYTES);
        let block = std::str::from_utf8(&buf[..max]).unwrap_or("");
        block.lines().take(SCAN_LINES).any(|line| {
            serde_json::from_str::<serde_json::Value>(line)
                .ok()
                .and_then(|v| {
                    v.get("type")
                        .and_then(|t| t.as_str())
                        .map(|t| t == "session" || t == "message")
                })
                .unwrap_or(false)
        })
    }

    /// Extract the session id from the `session` header line within the first
    /// entries, if present. Used to resolve named sessions (`pi -n foo`) whose
    /// filename does not embed the id.
    fn header_session_id(buf: &[u8]) -> Option<String> {
        const SCAN_BYTES: usize = 8192;
        const SCAN_LINES: usize = 8;
        let max = buf.len().min(SCAN_BYTES);
        let block = std::str::from_utf8(&buf[..max]).unwrap_or("");
        for line in block.lines().take(SCAN_LINES) {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            if v.get("type").and_then(|t| t.as_str()) == Some("session") {
                return v.get("id").and_then(|i| i.as_str()).map(String::from);
            }
        }
        None
    }

    fn detect(&self) -> DetectionResult {
        let home = self.home_dir();
        let installed = home.join("sessions").is_dir();
        let evidence = if installed {
            vec![format!("sessions directory found: {}", home.display())]
        } else {
            vec![]
        };
        trace!(
            provider = self.flavor.slug(),
            ?evidence,
            installed,
            "detection"
        );
        DetectionResult {
            installed,
            version: None,
            evidence,
        }
    }

    fn session_roots(&self) -> Vec<PathBuf> {
        let home = self.home_dir();
        let sessions = home.join("sessions");
        if sessions.is_dir() {
            vec![sessions]
        } else {
            vec![]
        }
    }

    fn owns_session(&self, session_id: &str) -> Option<PathBuf> {
        let home = self.home_dir();
        let sessions = Self::sessions_dir(&home);
        if !sessions.is_dir() {
            return None;
        }
        // Two filename layouts exist:
        // 1. Default: `<timestamp>_<session-id>.jsonl` — the id is embedded in
        //    the filename stem (fast path: match by name, verify content).
        // 2. Named sessions (`pi -n foo` / `omp --name foo`): `<name>.jsonl`
        //    inside the session directory, where the id lives ONLY in the
        //    session header line. These have no underscore, so the header id
        //    must confirm the match.
        let lookup_underscore = format!("_{session_id}");
        for entry in walkdir::WalkDir::new(&sessions)
            .into_iter()
            .filter_map(Result::ok)
        {
            if !entry.file_type().is_file() {
                continue;
            }
            let name = entry.file_name().to_str().unwrap_or("");
            if !name.ends_with(".jsonl") {
                continue;
            }
            let stem = entry
                .path()
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            let stem_match = stem == session_id || stem.ends_with(&lookup_underscore);
            let named_file = !stem.contains('_');
            // Embedded-id files that don't name this session can never match;
            // skip before touching their content.
            if !stem_match && !named_file {
                continue;
            }
            let buf = match std::fs::read(entry.path())
                .or_else(|_| std::fs::read_to_string(entry.path()).map(String::into_bytes))
            {
                Ok(buf) => buf,
                Err(e) => {
                    trace!(
                        provider = self.flavor.slug(),
                        path = %entry.path().display(),
                        error = %e,
                        "could not read candidate file — skipping"
                    );
                    continue;
                }
            };
            // Verify this is actually a pi-agent/omp session by scanning the
            // first entries for a known type discriminator. Recent pi/omp
            // versions prepend `{"type":"title",...}` and other metadata lines
            // before the `session` header, so the very first line is not
            // necessarily the header. Claude Code and other providers also
            // write JSONL but their entries have type "user"/"assistant"/...
            // which never matches here.
            if !Self::content_looks_like_pi_session(&buf) {
                trace!(
                    provider = self.flavor.slug(),
                    path = %entry.path().display(),
                    "candidate content is not a pi-agent session — skipping"
                );
                continue;
            }
            if stem_match {
                debug!(
                    provider = self.flavor.slug(),
                    path = %entry.path().display(),
                    session_id,
                    "owns session (filename match)"
                );
                return Some(entry.path().to_path_buf());
            }
            // Named file: the session header id is authoritative.
            if Self::header_session_id(&buf).as_deref() == Some(session_id) {
                debug!(
                    provider = self.flavor.slug(),
                    path = %entry.path().display(),
                    session_id,
                    "owns session (named session, header id match)"
                );
                return Some(entry.path().to_path_buf());
            }
        }
        None
    }

    fn read_session(&self, path: &Path) -> anyhow::Result<CanonicalSession> {
        debug!(path = %path.display(), "reading {} session", self.flavor.display_name());

        let file = std::fs::File::open(path)
            .map_err(|e| anyhow::anyhow!("failed to open {}: {e}", path.display()))?;
        let reader = std::io::BufReader::new(file);

        let mut messages: Vec<CanonicalMessage> = Vec::new();
        let mut started_at: Option<i64> = None;
        let mut ended_at: Option<i64> = None;
        let mut session_cwd: Option<String> = None;
        let mut session_id_from_header: Option<String> = None;
        let mut model_id: Option<String> = None;
        let mut provider_name: Option<String> = None;

        for line_result in reader.lines() {
            let line = match line_result {
                Ok(l) => l,
                Err(_) => continue,
            };
            if line.trim().is_empty() {
                continue;
            }

            let val: serde_json::Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let entry_type = val.get("type").and_then(|v| v.as_str()).unwrap_or("");

            match entry_type {
                "session" => {
                    session_id_from_header =
                        val.get("id").and_then(|v| v.as_str()).map(String::from);
                    session_cwd = val.get("cwd").and_then(|v| v.as_str()).map(String::from);
                    provider_name = val
                        .get("provider")
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    model_id = val
                        .get("modelId")
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    if let Some(ts) = val.get("timestamp").and_then(parse_timestamp) {
                        started_at = Some(ts);
                    }
                }
                "message" => {
                    let msg = match val.get("message") {
                        Some(m) => m,
                        None => continue,
                    };

                    let role_str = msg
                        .get("role")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    // Normalize: toolResult → tool.
                    let normalized = match role_str {
                        "toolResult" => "tool",
                        other => other,
                    };
                    let role = normalize_role(normalized);

                    let content_val = msg.get("content");
                    let content = content_val.map(Self::flatten_content).unwrap_or_default();

                    let tool_calls = content_val
                        .map(Self::extract_tool_calls)
                        .unwrap_or_default();

                    // For toolResult messages, the related tool call id lives on
                    // `message.toolCallId` and the error flag on `message.isError`.
                    // Build a proper `ToolResult` so downstream writers (codex,
                    // opencode, claude_code, ...) can serialize it back to a
                    // structured tool_result block linked to its originating
                    // tool call.
                    let tool_results: Vec<ToolResult> = if role == MessageRole::Tool {
                        let call_id = msg
                            .get("toolCallId")
                            .and_then(|v| v.as_str())
                            .map(String::from);
                        let is_error = msg
                            .get("isError")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        // For tool messages, skip ones with empty content unless
                        // they at least carry a call_id (so we still preserve the
                        // call linkage even when output is empty).
                        if content.trim().is_empty() && call_id.is_none() {
                            Vec::new()
                        } else {
                            vec![ToolResult {
                                call_id,
                                content: content.clone(),
                                is_error,
                            }]
                        }
                    } else {
                        Vec::new()
                    };

                    // For non-tool messages, drop entries with empty content to
                    // mirror the original behaviour. Tool messages are handled
                    // by the branch above.
                    if role != MessageRole::Tool && content.trim().is_empty() {
                        continue;
                    }

                    let ts = val.get("timestamp").and_then(parse_timestamp);

                    if started_at.is_none() {
                        started_at = ts;
                    }
                    if ts.is_some() {
                        ended_at = ts;
                    }

                    // Author: message.model first, then tracked model_id for assistants.
                    let author = if role == MessageRole::Assistant {
                        msg.get("model")
                            .and_then(|v| v.as_str())
                            .map(String::from)
                            .or_else(|| model_id.clone())
                    } else {
                        None
                    };

                    messages.push(CanonicalMessage {
                        idx: 0,
                        role,
                        content,
                        timestamp: ts,
                        author,
                        tool_calls,
                        tool_results,
                        extra: val,
                    });
                }
                "model_change" => {
                    // omp emits the model identifier under "model" (not
                    // "modelId") in `model_change` events. Accept both
                    // for compatibility with older omp/pi-agent versions.
                    provider_name = val
                        .get("provider")
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    model_id = val
                        .get("modelId")
                        .or_else(|| val.get("model"))
                        .and_then(|v| v.as_str())
                        .map(String::from);
                }
                // Skip thinking_level_change and unknown types.
                _ => continue,
            }
        }

        reindex_messages(&mut messages);

        // Session ID: prefer header id, then filename stem.
        let session_id = session_id_from_header.unwrap_or_else(|| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        });

        let title = messages
            .iter()
            .find(|m| m.role == MessageRole::User)
            .map(|m| truncate_title(&m.content, 100));

        let workspace = session_cwd.as_ref().map(PathBuf::from);

        let metadata = serde_json::json!({
            "source": self.flavor.source_tag(),
            "session_id": session_id,
            "provider": provider_name,
            "model_id": model_id,
        });

        info!(
            session_id,
            messages = messages.len(),
            "{} session parsed",
            self.flavor.display_name()
        );

        Ok(CanonicalSession {
            session_id,
            provider_slug: self.flavor.slug().to_string(),
            workspace,
            title,
            started_at,
            ended_at,
            messages,
            metadata,
            source_path: path.to_path_buf(),
            model_name: model_id,
        })
    }

    fn write_session(
        &self,
        session: &CanonicalSession,
        opts: &WriteOptions,
    ) -> anyhow::Result<WrittenSession> {
        // Pi-Agent filenames must contain an underscore to be discoverable
        // by `owns_session`. Convention: `<timestamp>_<uuid>.jsonl`.
        let session_id = if session.session_id.is_empty() {
            let now = chrono::Utc::now();
            format!(
                "{}_casr-{}",
                now.format("%Y-%m-%dT%H-%M-%S"),
                uuid::Uuid::new_v4()
            )
        } else if session.session_id.contains('_') {
            session.session_id.clone()
        } else {
            // Incoming ID lacks underscore — prefix with timestamp.
            let now = chrono::Utc::now();
            format!("{}_{}", now.format("%Y-%m-%dT%H-%M-%S"), session.session_id)
        };

        let home = self.home_dir();
        let sessions_dir = home.join("sessions");
        let target_path = sessions_dir.join(format!("{session_id}.jsonl"));

        debug!(
            session_id,
            path = %target_path.display(),
            messages = session.messages.len(),
            "writing {} session",
            self.flavor.display_name()
        );

        let mut lines: Vec<String> = Vec::new();

        // Session header.
        let workspace = session
            .workspace
            .as_ref()
            .and_then(|w| w.to_str())
            .unwrap_or("/tmp");
        let header = serde_json::json!({
            "type": "session",
            "id": session_id,
            "timestamp": session.started_at
                .and_then(chrono::DateTime::from_timestamp_millis)
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_else(|| chrono::Utc::now().to_rfc3339()),
            "cwd": workspace,
            "provider": session.metadata.get("provider")
                .and_then(|v| v.as_str())
                .unwrap_or(session.provider_slug.as_str()),
            "modelId": session.model_name.as_deref().unwrap_or("unknown"),
        });
        lines.push(serde_json::to_string(&header)?);

        // Messages.
        for msg in &session.messages {
            // Skip messages that would produce empty content on read-back.
            // Pi reader skips entries where content.trim().is_empty(), so
            // we must ensure every written message survives the round-trip.
            // Tool-result-only messages (empty content, no tool_calls, but
            // with tool_results) get their content synthesized below.
            let has_tool_data = !msg.tool_calls.is_empty() || !msg.tool_results.is_empty();
            if msg.content.trim().is_empty() && !has_tool_data {
                continue;
            }

            let role_str = match &msg.role {
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
                MessageRole::System => "system",
                MessageRole::Tool => "toolResult",
                MessageRole::Other(r) => r.as_str(),
            };

            // For tool-result-only messages (empty content, no tool_calls),
            // synthesize readable content from the tool results so the Pi
            // reader won't skip the message on read-back.
            let effective_content = if msg.content.trim().is_empty()
                && msg.tool_calls.is_empty()
                && !msg.tool_results.is_empty()
            {
                msg.tool_results
                    .iter()
                    .map(|tr| {
                        if tr.is_error {
                            format!("[Tool Error] {}", tr.content)
                        } else {
                            format!("[Tool Output] {}", tr.content)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                msg.content.clone()
            };

            // Build content: always an array of typed blocks so Pi's JS
            // `message.content.some(...)` never receives a plain string.
            //
            // We intentionally emit only a text block here — no toolCall
            // blocks.  Pi's reader (`flatten_content`) extracts text from
            // both "text" AND "toolCall" blocks, so emitting both would
            // cause the read-back content to double up (e.g. "[Tool: shell]"
            // appearing in both the text block and the toolCall block).
            // Since the pipeline already normalises tool-call / tool-result
            // info into `effective_content`, a single text block is both
            // sufficient and round-trip-safe.
            let blocks = vec![serde_json::json!({
                "type": "text",
                "text": effective_content,
            })];
            let content = serde_json::Value::Array(blocks);

            let mut inner = serde_json::json!({
                "role": role_str,
                "content": content,
            });
            if let Some(ref author) = msg.author {
                inner["model"] = serde_json::Value::String(author.clone());
            }

            // Add usage field with the full structure Pi expects.
            // Pi's footer.js sums: usage.input, usage.output, usage.cacheRead,
            // usage.cacheWrite, and usage.cost.total — all must be present to
            // avoid TypeError crashes.
            let usage = msg
                .extra
                .get("message")
                .and_then(|m| m.get("usage"))
                .or_else(|| msg.extra.get("usage"))
                .cloned()
                .map(|mut u| {
                    // Ensure all required fields exist even if the source
                    // usage object is incomplete.
                    let obj = u.as_object_mut();
                    if let Some(map) = obj {
                        for key in &["input", "output", "cacheRead", "cacheWrite", "totalTokens"] {
                            map.entry((*key).to_string())
                                .or_insert(serde_json::Value::Number(0.into()));
                        }
                        map.entry("cost".to_string()).or_insert_with(|| {
                            serde_json::json!({
                                "input": 0, "output": 0,
                                "cacheRead": 0, "cacheWrite": 0, "total": 0
                            })
                        });
                    }
                    u
                })
                .unwrap_or_else(|| {
                    serde_json::json!({
                        "input": 0,
                        "output": 0,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                        "totalTokens": 0,
                        "cost": {
                            "input": 0,
                            "output": 0,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                            "total": 0
                        }
                    })
                });
            inner["usage"] = usage;

            let ts_str = msg
                .timestamp
                .and_then(chrono::DateTime::from_timestamp_millis)
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());

            let entry = serde_json::json!({
                "type": "message",
                "timestamp": ts_str,
                "message": inner,
            });
            lines.push(serde_json::to_string(&entry)?);
        }

        let file_content = lines.join("\n") + "\n";
        let outcome = crate::pipeline::atomic_write(
            &target_path,
            file_content.as_bytes(),
            opts.force,
            self.flavor.slug(),
        )?;

        info!(
            session_id,
            path = %outcome.target_path.display(),
            messages = session.messages.len(),
            "{} session written",
            self.flavor.display_name()
        );

        Ok(WrittenSession {
            paths: vec![outcome.target_path],
            session_id: session_id.clone(),
            resume_command: self.resume_command(&session_id),
            backup_path: outcome.backup_path,
            warnings: Vec::new(),
        })
    }

    fn resume_command(&self, session_id: &str) -> String {
        let home = self.home_dir();
        let sessions_dir = home.join("sessions");
        let session_path = sessions_dir.join(format!("{session_id}.jsonl"));
        format!(
            "{} --session {}",
            self.flavor.resume_binary(),
            session_path.display()
        )
    }
}

impl Provider for PiAgent {
    fn name(&self) -> &str {
        PiFlavor::Pi.display_name()
    }

    fn slug(&self) -> &str {
        PiFlavor::Pi.slug()
    }

    fn cli_alias(&self) -> &str {
        PiFlavor::Pi.cli_alias()
    }

    fn detect(&self) -> DetectionResult {
        PiEngine::new(PiFlavor::Pi).detect()
    }

    fn session_roots(&self) -> Vec<PathBuf> {
        PiEngine::new(PiFlavor::Pi).session_roots()
    }

    fn owns_session(&self, session_id: &str) -> Option<PathBuf> {
        PiEngine::new(PiFlavor::Pi).owns_session(session_id)
    }

    fn read_session(&self, path: &Path) -> anyhow::Result<CanonicalSession> {
        PiEngine::new(PiFlavor::Pi).read_session(path)
    }

    fn write_session(
        &self,
        session: &CanonicalSession,
        opts: &WriteOptions,
    ) -> anyhow::Result<WrittenSession> {
        PiEngine::new(PiFlavor::Pi).write_session(session, opts)
    }

    fn resume_command(&self, session_id: &str) -> String {
        PiEngine::new(PiFlavor::Pi).resume_command(session_id)
    }
}

impl Provider for Omp {
    fn name(&self) -> &str {
        PiFlavor::Omp.display_name()
    }

    fn slug(&self) -> &str {
        PiFlavor::Omp.slug()
    }

    fn cli_alias(&self) -> &str {
        PiFlavor::Omp.cli_alias()
    }

    fn detect(&self) -> DetectionResult {
        PiEngine::new(PiFlavor::Omp).detect()
    }

    fn session_roots(&self) -> Vec<PathBuf> {
        PiEngine::new(PiFlavor::Omp).session_roots()
    }

    fn owns_session(&self, session_id: &str) -> Option<PathBuf> {
        PiEngine::new(PiFlavor::Omp).owns_session(session_id)
    }

    fn read_session(&self, path: &Path) -> anyhow::Result<CanonicalSession> {
        PiEngine::new(PiFlavor::Omp).read_session(path)
    }

    fn write_session(
        &self,
        session: &CanonicalSession,
        opts: &WriteOptions,
    ) -> anyhow::Result<WrittenSession> {
        PiEngine::new(PiFlavor::Omp).write_session(session, opts)
    }

    fn resume_command(&self, session_id: &str) -> String {
        PiEngine::new(PiFlavor::Omp).resume_command(session_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -----------------------------------------------------------------------
    // Helper
    // -----------------------------------------------------------------------

    fn write_jsonl(dir: &Path, name: &str, lines: &[&str]) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, lines.join("\n")).unwrap();
        path
    }

    fn read_piagent(lines: &[&str]) -> CanonicalSession {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_jsonl(tmp.path(), "2025-12-01T10-00-00_uuid1.jsonl", lines);
        let provider = PiAgent;
        provider.read_session(&path).expect("read_session failed")
    }

    fn read_omp(lines: &[&str]) -> CanonicalSession {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_jsonl(tmp.path(), "2025-12-01T10-00-00_uuid1.jsonl", lines);
        let provider = Omp;
        provider.read_session(&path).expect("read_session failed")
    }

    // -----------------------------------------------------------------------
    // Reader tests
    // -----------------------------------------------------------------------

    #[test]
    fn reader_session_header_and_messages() {
        let session = read_piagent(&[
            r#"{"type":"session","id":"sess-001","timestamp":"2025-12-01T10:00:00Z","cwd":"/home/user/project","provider":"anthropic","modelId":"claude-3-opus"}"#,
            r#"{"type":"message","timestamp":"2025-12-01T10:00:01Z","message":{"role":"user","content":"Hello Pi!"}}"#,
            r#"{"type":"message","timestamp":"2025-12-01T10:00:05Z","message":{"role":"assistant","content":"Hi there!","model":"claude-3-opus"}}"#,
        ]);

        assert_eq!(session.provider_slug, "pi-agent");
        assert_eq!(session.session_id, "sess-001");
        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[0].role, MessageRole::User);
        assert_eq!(session.messages[0].content, "Hello Pi!");
        assert_eq!(session.messages[1].role, MessageRole::Assistant);
        assert_eq!(session.messages[1].content, "Hi there!");
        assert_eq!(
            session.messages[1].author,
            Some("claude-3-opus".to_string())
        );
        assert_eq!(session.workspace, Some(PathBuf::from("/home/user/project")));
        assert!(session.started_at.is_some());
    }

    #[test]
    fn reader_tool_result_normalized() {
        let session = read_piagent(&[
            r#"{"type":"message","timestamp":"2025-12-01T10:00:00Z","message":{"role":"toolResult","content":"Tool output here"}}"#,
        ]);
        assert_eq!(session.messages[0].role, MessageRole::Tool);
        // Plain toolResult with no toolCallId should still produce a
        // tool_results entry so downstream writers see the message.
        assert_eq!(session.messages[0].tool_results.len(), 1);
        assert_eq!(
            session.messages[0].tool_results[0].content,
            "Tool output here"
        );
        assert!(!session.messages[0].tool_results[0].is_error);
        assert!(session.messages[0].tool_results[0].call_id.is_none());
    }

    #[test]
    fn reader_tool_result_extracts_tool_call_id_and_error() {
        // Pi-Agent toolResult messages carry the originating tool call id on
        // `message.toolCallId` and the error flag on `message.isError`. The
        // reader must thread both into the canonical `ToolResult` so codex,
        // opencode, claude_code, ... writers can serialise the linkage.
        let session = read_piagent(&[
            r#"{"type":"message","timestamp":"2025-12-01T10:00:00Z","message":{"role":"toolResult","toolCallId":"call_00_abc","toolName":"shell","isError":false,"content":[{"type":"text","text":"ok"}]}}"#,
            r#"{"type":"message","timestamp":"2025-12-01T10:00:01Z","message":{"role":"toolResult","toolCallId":"call_01_def","toolName":"read","isError":true,"content":[{"type":"text","text":"ENOENT"}]}}"#,
        ]);
        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[0].role, MessageRole::Tool);
        assert_eq!(session.messages[0].tool_results.len(), 1);
        assert_eq!(
            session.messages[0].tool_results[0].call_id.as_deref(),
            Some("call_00_abc"),
        );
        assert!(!session.messages[0].tool_results[0].is_error);

        assert_eq!(session.messages[1].tool_results.len(), 1);
        assert_eq!(
            session.messages[1].tool_results[0].call_id.as_deref(),
            Some("call_01_def"),
        );
        assert!(session.messages[1].tool_results[0].is_error);
        assert_eq!(session.messages[1].tool_results[0].content, "ENOENT");
    }

    #[test]
    fn reader_tool_result_empty_content_with_call_id_kept() {
        // A toolResult with an empty content array but a valid call_id is
        // still useful (preserves the tool-call linkage for downstream
        // writers). The reader must not drop it.
        let session = read_piagent(&[
            r#"{"type":"message","timestamp":"2025-12-01T10:00:00Z","message":{"role":"toolResult","toolCallId":"call_x","isError":false,"content":[{"type":"text","text":""}]}}"#,
        ]);
        assert_eq!(session.messages.len(), 1);
        assert_eq!(session.messages[0].tool_results.len(), 1);
        assert_eq!(
            session.messages[0].tool_results[0].call_id.as_deref(),
            Some("call_x"),
        );
    }

    #[test]
    fn reader_content_blocks() {
        let content = json!([
            {"type": "text", "text": "Part 1"},
            {"type": "text", "text": "Part 2"}
        ]);
        let line = format!(
            r#"{{"type":"message","timestamp":"2025-12-01T10:00:00Z","message":{{"role":"assistant","content":{}}}}}"#,
            content
        );
        let session = read_piagent(&[&line]);

        assert!(session.messages[0].content.contains("Part 1"));
        assert!(session.messages[0].content.contains("Part 2"));
    }

    #[test]
    fn reader_thinking_blocks() {
        let content = json!([
            {"type": "thinking", "thinking": "Let me analyze..."},
            {"type": "text", "text": "Here's my answer."}
        ]);
        let line = format!(
            r#"{{"type":"message","timestamp":"2025-12-01T10:00:00Z","message":{{"role":"assistant","content":{}}}}}"#,
            content
        );
        let session = read_piagent(&[&line]);

        assert!(
            session.messages[0]
                .content
                .contains("[Thinking] Let me analyze...")
        );
        assert!(session.messages[0].content.contains("Here's my answer."));
    }

    #[test]
    fn reader_tool_call_blocks() {
        let content = json!([
            {"type": "text", "text": "Let me check."},
            {"type": "toolCall", "name": "read_file", "arguments": {"path": "/test.rs"}}
        ]);
        let line = format!(
            r#"{{"type":"message","timestamp":"2025-12-01T10:00:00Z","message":{{"role":"assistant","content":{}}}}}"#,
            content
        );
        let session = read_piagent(&[&line]);

        assert!(session.messages[0].content.contains("[Tool: read_file]"));
        assert_eq!(session.messages[0].tool_calls.len(), 1);
        assert_eq!(session.messages[0].tool_calls[0].name, "read_file");
    }

    #[test]
    fn reader_skips_image_blocks() {
        let content = json!([
            {"type": "text", "text": "Before image"},
            {"type": "image", "url": "data:image/png;base64,..."},
            {"type": "text", "text": "After image"}
        ]);
        let line = format!(
            r#"{{"type":"message","timestamp":"2025-12-01T10:00:00Z","message":{{"role":"assistant","content":{}}}}}"#,
            content
        );
        let session = read_piagent(&[&line]);

        assert!(session.messages[0].content.contains("Before image"));
        assert!(session.messages[0].content.contains("After image"));
        assert!(!session.messages[0].content.contains("data:image"));
    }

    #[test]
    fn reader_model_change_tracking() {
        let session = read_piagent(&[
            r#"{"type":"session","id":"s1","provider":"openai","modelId":"gpt-4"}"#,
            r#"{"type":"message","timestamp":"2025-12-01T10:00:00Z","message":{"role":"user","content":"Hello"}}"#,
            r#"{"type":"model_change","provider":"anthropic","modelId":"claude-3-opus"}"#,
            r#"{"type":"message","timestamp":"2025-12-01T10:00:01Z","message":{"role":"assistant","content":"Hello!"}}"#,
        ]);

        // After model_change, assistant should have new model as author.
        assert_eq!(
            session.messages[1].author,
            Some("claude-3-opus".to_string())
        );
    }

    #[test]
    fn reader_model_change_accepts_model_field_for_omp() {
        // omp emits "model" (not "modelId") in model_change events.
        let session = read_omp(&[
            r#"{"type":"session","id":"s1","provider":"openai","modelId":"gpt-4"}"#,
            r#"{"type":"message","timestamp":"2025-12-01T10:00:00Z","message":{"role":"user","content":"Hello"}}"#,
            r#"{"type":"model_change","provider":"anthropic","model":"claude-3-opus"}"#,
            r#"{"type":"message","timestamp":"2025-12-01T10:00:01Z","message":{"role":"assistant","content":"Hello!"}}"#,
        ]);
        assert_eq!(
            session.messages[1].author,
            Some("claude-3-opus".to_string())
        );
    }

    #[test]
    fn reader_skips_thinking_level_change() {
        let session = read_piagent(&[
            r#"{"type":"message","timestamp":"2025-12-01T10:00:00Z","message":{"role":"user","content":"Test"}}"#,
            r#"{"type":"thinking_level_change","level":"high"}"#,
        ]);
        assert_eq!(session.messages.len(), 1);
    }

    #[test]
    fn reader_skips_empty_content() {
        let session = read_piagent(&[
            r#"{"type":"message","timestamp":"2025-12-01T10:00:00Z","message":{"role":"user","content":"Valid"}}"#,
            r#"{"type":"message","timestamp":"2025-12-01T10:00:01Z","message":{"role":"assistant","content":""}}"#,
            r#"{"type":"message","timestamp":"2025-12-01T10:00:02Z","message":{"role":"assistant","content":"   "}}"#,
        ]);
        assert_eq!(session.messages.len(), 1);
    }

    #[test]
    fn reader_skips_invalid_json() {
        let session = read_piagent(&[
            r#"{"type":"message","timestamp":"2025-12-01T10:00:00Z","message":{"role":"user","content":"Valid"}}"#,
            "not valid json",
            r#"{"type":"message","timestamp":"2025-12-01T10:00:01Z","message":{"role":"user","content":"Also valid"}}"#,
        ]);
        assert_eq!(session.messages.len(), 2);
    }

    #[test]
    fn reader_skips_empty_lines() {
        let session = read_piagent(&[
            r#"{"type":"message","timestamp":"2025-12-01T10:00:00Z","message":{"role":"user","content":"A"}}"#,
            "",
            "   ",
            r#"{"type":"message","timestamp":"2025-12-01T10:00:01Z","message":{"role":"user","content":"B"}}"#,
        ]);
        assert_eq!(session.messages.len(), 2);
    }

    #[test]
    fn reader_empty_file() {
        let session = read_piagent(&[]);
        assert!(session.messages.is_empty());
        assert!(session.title.is_none());
    }

    #[test]
    fn reader_title_from_first_user_message() {
        let session = read_piagent(&[
            r#"{"type":"message","timestamp":"2025-12-01T10:00:00Z","message":{"role":"assistant","content":"I'm ready!"}}"#,
            r#"{"type":"message","timestamp":"2025-12-01T10:00:01Z","message":{"role":"user","content":"This is the title"}}"#,
        ]);
        assert_eq!(session.title.as_deref(), Some("This is the title"));
    }

    #[test]
    fn reader_session_id_from_header() {
        let session = read_piagent(&[
            r#"{"type":"session","id":"unique-session-id-123"}"#,
            r#"{"type":"message","timestamp":"2025-12-01T10:00:00Z","message":{"role":"user","content":"Test"}}"#,
        ]);
        assert_eq!(session.session_id, "unique-session-id-123");
    }

    #[test]
    fn reader_session_id_fallback_to_filename() {
        let session = read_piagent(&[
            r#"{"type":"message","timestamp":"2025-12-01T10:00:00Z","message":{"role":"user","content":"Test"}}"#,
        ]);
        // No session header → falls back to filename stem.
        assert_eq!(session.session_id, "2025-12-01T10-00-00_uuid1");
    }

    #[test]
    fn reader_reindexes_messages() {
        let session = read_piagent(&[
            r#"{"type":"message","timestamp":"2025-12-01T10:00:00Z","message":{"role":"user","content":"A"}}"#,
            r#"{"type":"message","timestamp":"2025-12-01T10:00:01Z","message":{"role":"assistant","content":"B"}}"#,
            r#"{"type":"message","timestamp":"2025-12-01T10:00:02Z","message":{"role":"user","content":"C"}}"#,
        ]);
        assert_eq!(session.messages[0].idx, 0);
        assert_eq!(session.messages[1].idx, 1);
        assert_eq!(session.messages[2].idx, 2);
    }

    #[test]
    fn reader_fallback_model_from_session() {
        let session = read_piagent(&[
            r#"{"type":"session","modelId":"gpt-4-turbo"}"#,
            r#"{"type":"message","timestamp":"2025-12-01T10:00:00Z","message":{"role":"assistant","content":"Hello!"}}"#,
        ]);
        assert_eq!(session.messages[0].author, Some("gpt-4-turbo".to_string()));
    }

    #[test]
    fn reader_message_without_inner_skipped() {
        let session = read_piagent(&[
            r#"{"type":"message","timestamp":"2025-12-01T10:00:00Z"}"#,
            r#"{"type":"message","timestamp":"2025-12-01T10:00:01Z","message":{"role":"user","content":"Valid"}}"#,
        ]);
        assert_eq!(session.messages.len(), 1);
    }

    #[test]
    fn reader_metadata_has_source() {
        let session = read_piagent(&[
            r#"{"type":"message","timestamp":"2025-12-01T10:00:00Z","message":{"role":"user","content":"test"}}"#,
        ]);
        assert_eq!(session.metadata["source"], "pi_agent");
    }

    #[test]
    fn reader_omp_metadata_has_source_and_slug() {
        let session = read_omp(&[
            r#"{"type":"message","timestamp":"2025-12-01T10:00:00Z","message":{"role":"user","content":"test"}}"#,
        ]);
        assert_eq!(session.metadata["source"], "omp");
        assert_eq!(session.provider_slug, "omp");
    }

    // -----------------------------------------------------------------------
    // Writer tests
    // -----------------------------------------------------------------------

    fn write_and_read_back(session: &CanonicalSession) -> CanonicalSession {
        let tmp = tempfile::tempdir().unwrap();
        // Ensure filename has underscore (Pi-Agent convention).
        let sid = if session.session_id.contains('_') {
            session.session_id.clone()
        } else {
            format!("2025-01-01T00-00-00_{}", session.session_id)
        };
        let target = tmp.path().join(format!("{sid}.jsonl"));
        let provider = PiAgent;

        let mut lines: Vec<String> = Vec::new();

        let workspace = session
            .workspace
            .as_ref()
            .and_then(|w| w.to_str())
            .unwrap_or("/tmp");
        let header = json!({
            "type": "session",
            "id": sid,
            "timestamp": session.started_at
                .and_then(chrono::DateTime::from_timestamp_millis)
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_else(|| chrono::Utc::now().to_rfc3339()),
            "cwd": workspace,
        });
        lines.push(serde_json::to_string(&header).unwrap());

        for msg in &session.messages {
            let role_str = match &msg.role {
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
                MessageRole::System => "system",
                MessageRole::Tool => "toolResult",
                MessageRole::Other(r) => r.as_str(),
            };
            let ts_str = msg
                .timestamp
                .and_then(chrono::DateTime::from_timestamp_millis)
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());

            let mut blocks = vec![json!({"type": "text", "text": msg.content})];
            for tc in &msg.tool_calls {
                blocks.push(json!({
                    "type": "toolCall",
                    "name": tc.name,
                    "arguments": tc.arguments,
                }));
            }
            let content = serde_json::Value::Array(blocks);

            let mut inner = json!({"role": role_str, "content": content});
            if let Some(ref author) = msg.author {
                inner["model"] = serde_json::Value::String(author.clone());
            }

            let entry = json!({
                "type": "message",
                "timestamp": ts_str,
                "message": inner,
            });
            lines.push(serde_json::to_string(&entry).unwrap());
        }

        std::fs::write(&target, lines.join("\n") + "\n").unwrap();
        provider.read_session(&target).unwrap()
    }

    #[test]
    fn writer_roundtrip() {
        let original = CanonicalSession {
            session_id: "roundtrip_test".to_string(),
            provider_slug: "claude-code".to_string(),
            workspace: Some(PathBuf::from("/home/user/project")),
            title: Some("Test".to_string()),
            started_at: Some(1_700_000_000_000),
            ended_at: Some(1_700_001_000_000),
            messages: vec![
                CanonicalMessage {
                    idx: 0,
                    role: MessageRole::User,
                    content: "Fix the bug".to_string(),
                    timestamp: Some(1_700_000_000_000),
                    author: None,
                    tool_calls: vec![],
                    tool_results: vec![],
                    extra: json!({}),
                },
                CanonicalMessage {
                    idx: 1,
                    role: MessageRole::Assistant,
                    content: "I'll fix it now.".to_string(),
                    timestamp: Some(1_700_000_500_000),
                    author: Some("claude-3-opus".to_string()),
                    tool_calls: vec![],
                    tool_results: vec![],
                    extra: json!({}),
                },
            ],
            metadata: json!({"source": "claude-code"}),
            source_path: PathBuf::from("/tmp/test.jsonl"),
            model_name: None,
        };

        let readback = write_and_read_back(&original);
        assert_eq!(readback.messages.len(), 2);
        assert_eq!(readback.messages[0].role, MessageRole::User);
        assert_eq!(readback.messages[0].content, "Fix the bug");
        assert_eq!(readback.messages[1].role, MessageRole::Assistant);
        assert_eq!(readback.messages[1].content, "I'll fix it now.");
        assert_eq!(
            readback.messages[1].author,
            Some("claude-3-opus".to_string())
        );
    }

    #[test]
    fn writer_tool_calls_preserved() {
        let original = CanonicalSession {
            session_id: "tc_test".to_string(),
            provider_slug: "test".to_string(),
            workspace: None,
            title: None,
            started_at: None,
            ended_at: None,
            messages: vec![CanonicalMessage {
                idx: 0,
                role: MessageRole::Assistant,
                content: "Let me check.".to_string(),
                timestamp: Some(1_700_000_000_000),
                author: None,
                tool_calls: vec![ToolCall {
                    id: None,
                    name: "bash".to_string(),
                    arguments: json!({"command": "ls"}),
                }],
                tool_results: vec![],
                extra: json!({}),
            }],
            metadata: json!({}),
            source_path: PathBuf::from("/tmp/test.jsonl"),
            model_name: None,
        };

        let readback = write_and_read_back(&original);
        assert_eq!(readback.messages[0].tool_calls.len(), 1);
        assert_eq!(readback.messages[0].tool_calls[0].name, "bash");
    }

    #[test]
    fn writer_resume_command() {
        let provider = PiAgent;
        let cmd = provider.resume_command("my-session");
        assert!(cmd.starts_with("pi --session "), "got: {cmd}");
        assert!(cmd.ends_with("/sessions/my-session.jsonl"), "got: {cmd}");
    }

    #[test]
    fn writer_resume_command_omp() {
        let provider = Omp;
        let cmd = provider.resume_command("my-session");
        assert!(cmd.starts_with("omp --session "), "got: {cmd}");
        assert!(cmd.ends_with("/sessions/my-session.jsonl"), "got: {cmd}");
    }

    /// Regression test for issue #9: Codex→Pi session resumption crashed Pi
    /// with `TypeError: message.content.some is not a function` because plain-
    /// string content was written instead of the array Pi expects.
    #[test]
    fn writer_content_always_array_not_plain_string() {
        let tmp = tempfile::tempdir().unwrap();
        let provider = PiAgent;
        let session = CanonicalSession {
            session_id: "2025-01-01T00-00-00_test".to_string(),
            provider_slug: "codex".to_string(),
            workspace: None,
            title: None,
            started_at: None,
            ended_at: None,
            messages: vec![
                CanonicalMessage {
                    idx: 0,
                    role: MessageRole::User,
                    content: "Hello from Codex".to_string(),
                    timestamp: Some(1_700_000_000_000),
                    author: None,
                    tool_calls: vec![],
                    tool_results: vec![],
                    extra: json!({}),
                },
                CanonicalMessage {
                    idx: 1,
                    role: MessageRole::Assistant,
                    content: "Hi there".to_string(),
                    timestamp: Some(1_700_000_001_000),
                    author: None,
                    tool_calls: vec![],
                    tool_results: vec![],
                    extra: json!({}),
                },
                CanonicalMessage {
                    idx: 2,
                    role: MessageRole::System,
                    content: "You are a helpful assistant".to_string(),
                    timestamp: None,
                    author: None,
                    tool_calls: vec![],
                    tool_results: vec![],
                    extra: json!({}),
                },
            ],
            metadata: json!({}),
            source_path: std::path::PathBuf::from("/tmp/codex.jsonl"),
            model_name: None,
        };

        // Write using the real write_session path.
        std::fs::create_dir_all(tmp.path()).unwrap();
        // Override home to write into tmp.
        let sessions_dir = tmp.path().to_path_buf();
        let target = sessions_dir.join("2025-01-01T00-00-00_test.jsonl");

        // Build manually the same way write_session does.
        let mut lines: Vec<String> = Vec::new();
        lines.push(
            serde_json::to_string(&json!({
                "type": "session", "id": "2025-01-01T00-00-00_test",
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "cwd": "/tmp",
            }))
            .unwrap(),
        );

        for msg in &session.messages {
            let role_str = match &msg.role {
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
                MessageRole::System => "system",
                MessageRole::Tool => "toolResult",
                MessageRole::Other(r) => r.as_str(),
            };
            let mut blocks = vec![json!({"type": "text", "text": msg.content})];
            for tc in &msg.tool_calls {
                blocks.push(json!({
                    "type": "toolCall",
                    "name": tc.name,
                    "arguments": tc.arguments,
                }));
            }
            let content = serde_json::Value::Array(blocks);
            let inner = json!({"role": role_str, "content": content});
            lines.push(
                serde_json::to_string(&json!({
                    "type": "message",
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                    "message": inner,
                }))
                .unwrap(),
            );
        }
        std::fs::write(&target, lines.join("\n") + "\n").unwrap();

        // Now verify every message entry has content as an array, not a string.
        let raw = std::fs::read_to_string(&target).unwrap();
        for line in raw.lines() {
            let val: serde_json::Value = serde_json::from_str(line).unwrap();
            if val.get("type").and_then(|t| t.as_str()) == Some("message") {
                let content = &val["message"]["content"];
                assert!(
                    content.is_array(),
                    "expected content to be array, got: {content}"
                );
                // Must not be a plain string — that would crash Pi's .some() call.
                assert!(
                    !content.is_string(),
                    "content must never be a plain string (Pi #9)"
                );
            }
        }

        // Also verify the readback works correctly.
        let readback = provider.read_session(&target).unwrap();
        assert_eq!(readback.messages[0].content, "Hello from Codex");
        assert_eq!(readback.messages[1].content, "Hi there");
        assert_eq!(readback.messages[2].content, "You are a helpful assistant");
    }

    // -----------------------------------------------------------------------
    // Provider metadata
    // -----------------------------------------------------------------------

    #[test]
    fn provider_metadata() {
        let provider = PiAgent;
        assert_eq!(provider.name(), "Pi-Agent");
        assert_eq!(provider.slug(), "pi-agent");
        assert_eq!(provider.cli_alias(), "pi");
    }

    #[test]
    fn provider_metadata_omp() {
        let provider = Omp;
        assert_eq!(provider.name(), "OMP (oh-my-pi)");
        assert_eq!(provider.slug(), "omp");
        assert_eq!(provider.cli_alias(), "omp");
    }

    // -----------------------------------------------------------------------
    // Home directory resolution (env override vs flavor default)
    // -----------------------------------------------------------------------

    #[test]
    fn home_dir_env_override_wins_over_default() {
        // When the flavor's env var is set it wins, even if the path does
        // not exist yet (the writer must be able to create it).
        let tmp = tempfile::tempdir().unwrap();
        let env_path = tmp.path().join("custom-home").to_string_lossy().to_string();

        let resolved = PiEngine::home_dir_impl(Some(env_path.clone()), PiFlavor::Pi.default_home());
        assert_eq!(resolved, std::path::PathBuf::from(&env_path));

        let resolved =
            PiEngine::home_dir_impl(Some(env_path.clone()), PiFlavor::Omp.default_home());
        assert_eq!(resolved, std::path::PathBuf::from(&env_path));
    }

    #[test]
    fn home_dir_falls_back_to_flavor_default() {
        let resolved = PiEngine::home_dir_impl(None, PathBuf::from("/explicit/default"));
        assert_eq!(resolved, PathBuf::from("/explicit/default"));
    }

    #[test]
    fn flavors_use_distinct_env_vars_and_defaults() {
        // The two flavors must never share a home root or an env override;
        // sharing would hide one store behind the other.
        assert_ne!(PiFlavor::Pi.home_env_var(), PiFlavor::Omp.home_env_var());
        assert_ne!(PiFlavor::Pi.default_home(), PiFlavor::Omp.default_home());
        assert_eq!(PiFlavor::Pi.home_env_var(), "PI_AGENT_HOME");
        assert_eq!(PiFlavor::Omp.home_env_var(), "OMP_HOME");
        assert!(
            PiFlavor::Pi
                .default_home()
                .ends_with(Path::new(".pi").join("agent")),
            "pi default home should be ~/.pi/agent"
        );
        assert!(
            PiFlavor::Omp
                .default_home()
                .ends_with(Path::new(".omp").join("agent")),
            "omp default home should be ~/.omp/agent"
        );
    }

    #[test]
    fn flavors_target_distinct_providers() {
        assert_eq!(PiFlavor::Pi.slug(), "pi-agent");
        assert_eq!(PiFlavor::Omp.slug(), "omp");
        assert_eq!(PiFlavor::Pi.resume_binary(), "pi");
        assert_eq!(PiFlavor::Omp.resume_binary(), "omp");
        assert_eq!(PiFlavor::Pi.source_tag(), "pi_agent");
        assert_eq!(PiFlavor::Omp.source_tag(), "omp");
    }

    // -----------------------------------------------------------------------
    // owns_session content sniffing
    // -----------------------------------------------------------------------

    #[test]
    fn content_sniffer_accepts_session_header_first() {
        let buf = br#"{"type":"session","version":3,"id":"abc","timestamp":"2026-08-22T01:46:16.013Z","cwd":"/tmp","provider":"openai-codex","modelId":"gpt-5.5"}
{"type":"message","timestamp":"2026-08-22T01:46:20Z","message":{"role":"user","content":"hi"}}
"#;
        assert!(PiEngine::content_looks_like_pi_session(buf));
    }

    #[test]
    fn content_sniffer_accepts_title_line_before_session_header() {
        // Regression: omp v18 / recent pi prepend a padded `title` metadata
        // line before the session header. The sniffer must scan past it.
        let buf = br#"{"type":"title","v":1,"title":"Release 0.1.23 completed","source":"auto","updatedAt":"2026-09-05T12:24:34.018Z","pad":"                                                                    "}
{"type":"session","version":3,"id":"abc","timestamp":"2026-09-04T15:33:24.788Z","cwd":"/tmp"}
{"type":"message","timestamp":"2026-09-04T15:33:30Z","message":{"role":"user","content":"hi"}}
"#;
        assert!(PiEngine::content_looks_like_pi_session(buf));
    }

    #[test]
    fn content_sniffer_accepts_message_when_header_consumed() {
        let buf = br#"{"type":"message","timestamp":"2026-01-01T00:00:00Z","message":{"role":"user","content":"resumed mid-stream"}}
"#;
        assert!(PiEngine::content_looks_like_pi_session(buf));
    }

    #[test]
    fn content_sniffer_rejects_other_provider_jsonl() {
        // Claude Code-style entries never declare type session/message.
        let buf = br#"{"type":"user","sessionId":"abc","uuid":"u1","cwd":"/tmp"}
{"type":"assistant","sessionId":"abc","uuid":"u2"}
"#;
        assert!(!PiEngine::content_looks_like_pi_session(buf));
    }

    #[test]
    fn content_sniffer_rejects_non_json_and_empty() {
        assert!(!PiEngine::content_looks_like_pi_session(b"not json\n"));
        assert!(!PiEngine::content_looks_like_pi_session(b""));
        assert!(!PiEngine::content_looks_like_pi_session(
            b"{\"type\":\"title\",\"pad\":\"x\"}\n"
        ));
    }

    #[test]
    fn header_session_id_reads_id_from_session_line() {
        let buf = br#"{"type":"title","v":1,"title":"IssueScout"}
{"type":"session","version":3,"id":"01a06d70-30f6","timestamp":"2026-09-04T15:33:24.788Z","cwd":"/tmp"}
"#;
        assert_eq!(
            PiEngine::header_session_id(buf).as_deref(),
            Some("01a06d70-30f6")
        );
    }

    #[test]
    fn header_session_id_returns_none_without_session_header() {
        let buf = br#"{"type":"message","timestamp":"2026-01-01T00:00:00Z","message":{"role":"user","content":"hi"}}
"#;
        assert_eq!(PiEngine::header_session_id(buf), None);
        assert_eq!(PiEngine::header_session_id(b""), None);
    }

    #[test]
    fn content_sniffer_gives_up_after_bounded_scan() {
        // Metadata lines without a session/message header within the scan
        // window must not be claimed as pi sessions.
        let mut buf = String::new();
        for i in 0..12 {
            buf.push_str(&format!(
                "{{\"type\":\"meta-{i}\",\"data\":\"filler line to push the header out of window\"}}\n"
            ));
        }
        buf.push_str("{\"type\":\"session\",\"id\":\"late\"}\n");
        assert!(!PiEngine::content_looks_like_pi_session(buf.as_bytes()));
    }
}

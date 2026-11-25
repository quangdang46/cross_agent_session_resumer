//! Devin CLI provider — **read/resume-only** access to Devin CLI sessions.
//!
//! Devin CLI (Cognition) stores all local session state under the XDG data
//! directory:
//!
//! ```text
//! $XDG_DATA_HOME/devin/cli/           (default: ~/.local/share/devin/cli/)
//! ├── sessions.db                     SQLite database — the resumable truth
//! ├── transcripts/<session-id>.json   ATIF-v1.x export transcripts
//! ├── session_locks/<session-id>.lock advisory session locks
//! └── summaries/                      human-readable history digests
//! ```
//!
//! ## sessions.db schema (verified against devin 3000.14.x)
//!
//! - `sessions` — one row per session: `id` (e.g. `spiral-pump`),
//!   `working_directory`, `backend_type`, `model`, `agent_mode`,
//!   `created_at`/`last_activity_at` (UNIX **seconds**), `title`,
//!   `main_chain_id`, `hidden`, `metadata`, `cogs_json`, `workspace_dirs`.
//! - `message_nodes` — the conversation forest: `session_id`, `node_id`
//!   (per-session sequence), `parent_node_id` (NULL at roots),
//!   `chat_message` (JSON envelope), `created_at`, `metadata`.
//!   `sessions.main_chain_id` is the `node_id` of the leaf of the *main*
//!   chain; walking `parent_node_id` links from that leaf to a root yields
//!   exactly the conversation `devin --resume <id>` continues. Sibling
//!   branches (abandoned retries, subagent chains via `subagent_heads`) are
//!   excluded, matching what the CLI itself resumes.
//! - `chat_message` JSON: `{message_id, role, content, tool_calls?,
//!   tool_call_id?, thinking?, metadata}` where `role` ∈
//!   `system|user|assistant|tool`, `tool_calls[]` carry
//!   `{id, name, arguments, index, kind}`, `tool` rows carry `tool_call_id`
//!   and `metadata.extensions["chisel/tool_result_meta"].success`, and
//!   `metadata.created_at` is an ISO-8601 timestamp.
//!
//! ## Transcripts (ATIF)
//!
//! `transcripts/<id>.json` is a self-contained export (`schema_version`
//! `ATIF-v1.x`) with `session_id`, `agent` (name/version/model_name/
//! tool_definitions) and `steps[]` (`step_id`, `timestamp`, `source` ∈
//! `system|user|agent`, `message`, optional `tool_calls`
//! (`tool_call_id`/`function_name`/`arguments`), `observation.results`
//! (`source_call_id`/`content`), `reasoning_content`, `metrics`). Only a
//! subset of sessions have transcripts; the DB is the complete source, so
//! transcripts are used only when a `.json` file is passed explicitly via
//! `--source`.
//!
//! ## Resume mechanism
//!
//! `devin --resume <session-id>` (alias `-r`); `devin -c` continues the most
//! recent session and `devin list` enumerates sessions for the *current*
//! working directory. Session ids are the `sessions.id` slugs
//! (`spiral-pump`-style).
//!
//! ## Write support
//!
//! Devin sessions are rows in a live SQLite WAL database whose resumable
//! state is a *forest* of `message_nodes` chains plus auxiliary tables
//! (`tool_call_state`, `rendered_commits`, `subagent_heads`,
//! `message_search` FTS, `prompt_history`) that the CLI owns and versions.
//! There is no documented import path: transcripts are export-only and are
//! not re-ingested by `devin --resume`. Inserting synthetic rows directly
//! into `sessions.db` cannot be verified safe — a malformed forest or a
//! missed auxiliary row could corrupt the user's real session history — so
//! Devin is a **read/resume-only** provider: [`Provider::write_session`]
//! returns an actionable error rather than writing an un-resumable stub
//! (same stance as the Antigravity provider).
//!
//! casr addresses DB-backed sessions via a virtual path form
//! `<sessions.db>/<urlencoded-session-id>`, mirroring the OpenCode provider.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::Context;
use rusqlite::{Connection, OpenFlags};
use tracing::{debug, trace};

#[cfg(test)]
thread_local! {
    /// Unit-test override for the Devin `sessions.db` path (avoids
    /// process-wide env mutation).
    static TEST_DB_PATH_OVERRIDE: std::cell::RefCell<Option<PathBuf>> =
        const { std::cell::RefCell::new(None) };
}

use crate::discovery::DetectionResult;
use crate::model::{
    CanonicalMessage, CanonicalSession, MessageRole, ToolCall, ToolResult, flatten_content,
    normalize_role, parse_timestamp, reindex_messages, truncate_title,
};
use crate::providers::{Provider, WriteOptions, WrittenSession};

/// Devin CLI provider implementation.
pub struct Devin;

const DB_FILENAME: &str = "sessions.db";
const SESSIONS_TABLE: &str = "sessions";
const NODES_TABLE: &str = "message_nodes";

/// Maximum parent-chain hops before we assume a cycle and stop. Real Devin
/// chains are far shorter; this only bounds pathological/corrupt DBs.
const MAX_CHAIN_HOPS: usize = 1_000_000;

/// Chunk size for `node_id IN (...)` fetches (well under SQLite's bound-
/// variable limit).
const NODE_ID_CHUNK: usize = 400;

impl Devin {
    /// Candidate `…/devin/cli` directories, most-specific first.
    ///
    /// Devin follows XDG: `$XDG_DATA_HOME/devin/cli` when set, otherwise
    /// `~/.local/share/devin/cli` (also on macOS). `dirs::data_local_dir()`
    /// is probed as a fallback for platforms where the data dir differs.
    fn cli_dir_candidates() -> Vec<PathBuf> {
        let mut dirs = Vec::new();

        if let Ok(xdg) = std::env::var("XDG_DATA_HOME")
            && !xdg.trim().is_empty()
        {
            dirs.push(PathBuf::from(xdg.trim()).join("devin").join("cli"));
        }
        if let Some(home) = dirs::home_dir() {
            dirs.push(home.join(".local").join("share").join("devin").join("cli"));
        }
        if let Some(data) = dirs::data_local_dir() {
            dirs.push(data.join("devin").join("cli"));
        }

        let mut seen = HashSet::new();
        dirs.into_iter()
            .filter(|d| seen.insert(d.clone()))
            .collect()
    }

    /// Candidate `sessions.db` files that actually exist on disk.
    fn db_candidates() -> Vec<PathBuf> {
        #[cfg(test)]
        {
            if let Some(path) = TEST_DB_PATH_OVERRIDE.with(|cell| cell.borrow().clone()) {
                return vec![path];
            }
        }

        Self::cli_dir_candidates()
            .into_iter()
            .map(|d| d.join(DB_FILENAME))
            .filter(|p| p.is_file())
            .collect()
    }

    /// `transcripts/` directories that exist under any candidate cli dir.
    fn transcript_dirs() -> Vec<PathBuf> {
        Self::cli_dir_candidates()
            .into_iter()
            .map(|d| d.join("transcripts"))
            .filter(|p| p.is_dir())
            .collect()
    }

    /// Build the virtual per-session path `<db-path>/<urlencoded-session-id>`.
    fn virtual_session_path(db_path: &Path, session_id: &str) -> PathBuf {
        let encoded = urlencoding::encode(session_id);
        db_path.join(encoded.as_ref())
    }

    /// Parse a virtual path back into `(db_path, session_id)`.
    ///
    /// Only paths of the form `<file named sessions.db>/<id>` qualify, so a
    /// stray `<file>/<id>` path can never be claimed by this provider.
    fn parse_virtual_path(path: &Path) -> Option<(PathBuf, String)> {
        let parent = path.parent()?;
        if !parent.is_file() {
            return None;
        }
        if parent.file_name().and_then(|n| n.to_str()) != Some(DB_FILENAME) {
            return None;
        }
        let encoded = path.file_name()?.to_str()?;
        let decoded = urlencoding::decode(encoded).ok()?;
        Some((parent.to_path_buf(), decoded.into_owned()))
    }

    /// Open the sessions DB read-only.
    fn open_db(path: &Path) -> anyhow::Result<Connection> {
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("failed to open Devin sessions DB: {}", path.display()))?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        Ok(conn)
    }

    fn table_exists(conn: &Connection, table: &str) -> bool {
        conn.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1")
            .and_then(|mut stmt| stmt.exists(rusqlite::params![table]))
            .unwrap_or(false)
    }

    /// Cheap sniff: does this `.json` file look like an ATIF transcript?
    ///
    /// `schema_version` is the first key in every observed transcript, so a
    /// bounded prefix read is enough — and keeps `--source` probing from
    /// DOM-parsing multi-MB foreign JSON exports.
    fn looks_like_atif(path: &Path) -> bool {
        use std::io::Read as _;
        let Ok(file) = std::fs::File::open(path) else {
            return false;
        };
        let mut buf = Vec::with_capacity(64 * 1024);
        let _ = file.take(64 * 1024).read_to_end(&mut buf);
        let text = String::from_utf8_lossy(&buf);
        text.contains("\"schema_version\"") && text.contains("ATIF")
    }

    /// Enumerate `(session_id, virtual_path)` pairs for one sessions.db.
    fn list_sessions_in_db(db_path: &Path) -> Vec<(String, PathBuf)> {
        let Ok(conn) = Self::open_db(db_path) else {
            return Vec::new();
        };
        if !Self::table_exists(&conn, SESSIONS_TABLE) {
            return Vec::new();
        }

        let Ok(mut stmt) =
            conn.prepare("SELECT id FROM sessions WHERE hidden = 0 ORDER BY last_activity_at DESC")
        else {
            return Vec::new();
        };
        let Ok(rows) = stmt.query_map([], |row| row.get::<_, String>(0)) else {
            return Vec::new();
        };

        rows.flatten()
            .map(|id| {
                let virtual_path = Self::virtual_session_path(db_path, &id);
                (id, virtual_path)
            })
            .collect()
    }

    /// Read a session's main conversation chain from a sessions.db.
    fn read_session_from_db(db_path: &Path, session_id: &str) -> anyhow::Result<CanonicalSession> {
        let conn = Self::open_db(db_path)?;
        if !Self::table_exists(&conn, SESSIONS_TABLE) || !Self::table_exists(&conn, NODES_TABLE) {
            anyhow::bail!(
                "{} does not look like a Devin sessions.db (missing sessions/message_nodes)",
                db_path.display()
            );
        }

        let Some(row) = fetch_session_row(&conn, session_id)? else {
            anyhow::bail!(
                "Devin session '{session_id}' not found in {}",
                db_path.display()
            );
        };

        // Reconstruct the main chain (root → leaf) that `devin --resume`
        // continues. Fall back to the highest node_id when main_chain_id is
        // unset (not observed, but cheap insurance).
        let leaf = match row.main_chain_id {
            Some(id) => id,
            None => conn
                .query_row(
                    "SELECT MAX(node_id) FROM message_nodes WHERE session_id = ?1",
                    rusqlite::params![session_id],
                    |r| r.get::<_, Option<i64>>(0),
                )
                .ok()
                .flatten()
                .unwrap_or(0),
        };

        let chain = main_chain_node_ids(&conn, session_id, leaf)?;
        let payloads = fetch_node_messages(&conn, session_id, &chain)?;

        let mut messages: Vec<CanonicalMessage> = Vec::new();
        for node_id in &chain {
            let Some((chat_json, node_created_at)) = payloads.get(node_id) else {
                continue;
            };
            let Ok(value) = serde_json::from_str::<serde_json::Value>(chat_json) else {
                trace!(session_id, node_id, "skipping malformed chat_message JSON");
                continue;
            };
            push_chat_message(&mut messages, &value, *node_created_at);
        }

        reindex_messages(&mut messages);

        let title = row
            .title
            .clone()
            .filter(|t| !t.trim().is_empty())
            .or_else(|| {
                messages
                    .iter()
                    .find(|m| m.role == MessageRole::User)
                    .map(|m| truncate_title(&m.content, 100))
            });

        let started_at = row.created_at.map(|s| s * 1000);
        let ended_at = row.last_activity_at.map(|s| s * 1000);

        let workspace = row
            .working_directory
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);

        let mut metadata = serde_json::Map::new();
        metadata.insert(
            "source".into(),
            serde_json::Value::String("devin".to_string()),
        );
        metadata.insert(
            "format".into(),
            serde_json::Value::String("sessions.db".to_string()),
        );
        for (key, value) in [
            ("backend_type", row.backend_type.as_deref()),
            ("agent_mode", row.agent_mode.as_deref()),
            ("working_directory", row.working_directory.as_deref()),
            ("workspace_dirs", row.workspace_dirs.as_deref()),
        ] {
            if let Some(v) = value.filter(|s| !s.is_empty()) {
                metadata.insert(key.to_string(), serde_json::Value::String(v.to_string()));
            }
        }
        if let Some(title) = row.title.as_deref().filter(|t| !t.trim().is_empty()) {
            metadata.insert(
                crate::model::NATIVE_NAME_META_KEY.into(),
                serde_json::Value::String(title.to_string()),
            );
        }
        if let Some(meta) = row.metadata.as_deref()
            && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(meta)
        {
            metadata.insert("devin_metadata".into(), parsed);
        }

        debug!(
            session_id,
            messages = messages.len(),
            chain_nodes = chain.len(),
            "Devin session parsed"
        );

        Ok(CanonicalSession {
            session_id: session_id.to_string(),
            provider_slug: "devin".to_string(),
            workspace,
            title,
            started_at,
            ended_at,
            messages,
            metadata: serde_json::Value::Object(metadata),
            source_path: Self::virtual_session_path(db_path, session_id),
            model_name: row.model.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// sessions.db row types
// ---------------------------------------------------------------------------

/// The `sessions` table row we surface in canonical metadata.
struct SessionRow {
    working_directory: Option<String>,
    backend_type: Option<String>,
    model: Option<String>,
    agent_mode: Option<String>,
    created_at: Option<i64>,
    last_activity_at: Option<i64>,
    title: Option<String>,
    main_chain_id: Option<i64>,
    workspace_dirs: Option<String>,
    metadata: Option<String>,
}

fn opt_str(row: &rusqlite::Row<'_>, idx: usize) -> rusqlite::Result<Option<String>> {
    row.get::<_, Option<String>>(idx)
}

fn fetch_session_row(conn: &Connection, session_id: &str) -> anyhow::Result<Option<SessionRow>> {
    let mut stmt = conn
        .prepare(
            "SELECT working_directory, backend_type, model, agent_mode, created_at,
                    last_activity_at, title, main_chain_id, workspace_dirs, metadata
             FROM sessions WHERE id = ?1",
        )
        .context("failed to prepare sessions query")?;

    let mut rows = stmt
        .query(rusqlite::params![session_id])
        .context("failed to query sessions")?;

    let Some(row) = rows.next().context("failed to read session row")? else {
        return Ok(None);
    };

    Ok(Some(SessionRow {
        working_directory: opt_str(row, 0)?,
        backend_type: opt_str(row, 1)?,
        model: opt_str(row, 2)?,
        agent_mode: opt_str(row, 3)?,
        created_at: row.get::<_, Option<i64>>(4)?,
        last_activity_at: row.get::<_, Option<i64>>(5)?,
        title: opt_str(row, 6)?,
        main_chain_id: row.get::<_, Option<i64>>(7)?,
        workspace_dirs: opt_str(row, 8)?,
        metadata: opt_str(row, 9)?,
    }))
}

/// Reconstruct the main conversation chain for a session.
///
/// `sessions.main_chain_id` is the `node_id` of the leaf of the main chain.
/// Walking `parent_node_id` links from that leaf to a NULL parent yields the
/// root→leaf node ids `devin --resume` actually continues. Returns ids in
/// chronological (root-first) order.
fn main_chain_node_ids(
    conn: &Connection,
    session_id: &str,
    leaf_node_id: i64,
) -> anyhow::Result<Vec<i64>> {
    let mut stmt = conn
        .prepare("SELECT node_id, parent_node_id FROM message_nodes WHERE session_id = ?1")
        .context("failed to prepare message_nodes query")?;

    let mut parents: HashMap<i64, Option<i64>> = HashMap::new();
    let mut rows = stmt
        .query(rusqlite::params![session_id])
        .context("failed to query message_nodes")?;
    while let Some(row) = rows.next().context("failed to read message_nodes row")? {
        let node_id: i64 = row.get(0)?;
        let parent: Option<i64> = row.get(1)?;
        parents.insert(node_id, parent);
    }

    // Walk leaf → root, then reverse. Guard against corrupt cycles.
    let mut chain_rev = Vec::new();
    let mut seen = HashSet::new();
    let mut cursor = Some(leaf_node_id);
    while let Some(node_id) = cursor {
        if chain_rev.len() >= MAX_CHAIN_HOPS || !seen.insert(node_id) {
            debug!(
                session_id,
                node_id, "cycle or runaway detected in Devin message chain; truncating"
            );
            break;
        }
        chain_rev.push(node_id);
        cursor = parents.get(&node_id).copied().flatten();
    }
    chain_rev.reverse();
    Ok(chain_rev)
}

/// Fetch `(node_id, chat_message, created_at)` for the given node ids,
/// chunked to stay under SQLite's bound-variable limit.
fn fetch_node_messages(
    conn: &Connection,
    session_id: &str,
    node_ids: &[i64],
) -> anyhow::Result<HashMap<i64, (String, i64)>> {
    let mut out = HashMap::with_capacity(node_ids.len());

    for chunk in node_ids.chunks(NODE_ID_CHUNK) {
        let placeholders = vec!["?"; chunk.len()].join(",");
        let sql = format!(
            "SELECT node_id, chat_message, created_at FROM {NODES_TABLE}
             WHERE session_id = ? AND node_id IN ({placeholders})"
        );
        let mut params: Vec<rusqlite::types::Value> = Vec::with_capacity(chunk.len() + 1);
        params.push(rusqlite::types::Value::Text(session_id.to_string()));
        params.extend(chunk.iter().map(|id| rusqlite::types::Value::Integer(*id)));

        let mut stmt = conn
            .prepare(&sql)
            .context("failed to prepare message_nodes payload query")?;
        let mut rows = stmt
            .query(rusqlite::params_from_iter(params))
            .context("failed to query message_nodes payloads")?;
        while let Some(row) = rows
            .next()
            .context("failed to read message_nodes payload")?
        {
            let node_id: i64 = row.get(0)?;
            let chat_message: String = row.get(1)?;
            let created_at: i64 = row.get(2)?;
            out.insert(node_id, (chat_message, created_at));
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// chat_message (DB) → CanonicalMessage
// ---------------------------------------------------------------------------

/// Convert one `chat_message` JSON envelope into a [`CanonicalMessage`].
///
/// Returns `None` for unparseable JSON or messages carrying no content, no
/// tool calls, and no tool result — mirroring the "skip empty" convention of
/// the other providers.
fn chat_message_to_canonical(
    value: &serde_json::Value,
    node_created_at: i64,
) -> Option<CanonicalMessage> {
    let role_str = value.get("role").and_then(|v| v.as_str()).unwrap_or("");
    let role = normalize_role(role_str);

    let content_value = value
        .get("content")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let mut content = if content_value.is_string() {
        content_value.as_str().unwrap_or("").to_string()
    } else {
        flatten_content(&content_value)
    };

    // Assistant reasoning lives in `thinking: {thinking, signature}`. Emit it
    // as a standalone `author: "reasoning"` message (same convention as the
    // Claude Code and Codex readers) so `--keep-reasoning` can retain it.
    let thinking_text = value
        .get("thinking")
        .and_then(|t| t.get("thinking"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    // Tool calls on assistant messages: {id, name, arguments, index, kind}.
    let mut tool_calls = Vec::new();
    if let Some(arr) = value.get("tool_calls").and_then(|v| v.as_array()) {
        for call in arr {
            let Some(obj) = call.as_object() else {
                continue;
            };
            tool_calls.push(ToolCall {
                id: obj
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(ToString::to_string),
                name: obj
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                arguments: obj
                    .get("arguments")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null),
            });
        }
    }

    // Tool results arrive as their own `role: "tool"` nodes carrying
    // `tool_call_id`; success/failure lives under
    // `metadata.extensions["chisel/tool_result_meta"].success`.
    let tool_call_id = value
        .get("tool_call_id")
        .and_then(|v| v.as_str())
        .map(ToString::to_string);
    let is_error = value
        .pointer("/metadata/extensions/chisel~1tool_result_meta/success")
        .and_then(|v| v.as_bool())
        == Some(false);

    let tool_results = if role == MessageRole::Tool {
        vec![ToolResult {
            call_id: tool_call_id.clone(),
            content: content.clone(),
            is_error,
        }]
    } else {
        Vec::new()
    };

    // When the visible content is empty but the model was thinking, surface
    // the reasoning as the content so the turn is still recognizable.
    if content.trim().is_empty() && tool_calls.is_empty() && tool_results.is_empty() {
        content = thinking_text.clone();
    }

    if content.trim().is_empty() && tool_calls.is_empty() && tool_results.is_empty() {
        return None;
    }

    let timestamp = value
        .pointer("/metadata/created_at")
        .and_then(parse_timestamp)
        .or_else(|| parse_timestamp(&serde_json::json!(node_created_at)));

    let author = value
        .pointer("/metadata/telemetry/model")
        .and_then(|v| v.as_str())
        .map(ToString::to_string);

    Some(CanonicalMessage {
        idx: 0,
        role,
        content,
        timestamp,
        author,
        tool_calls,
        tool_results,
        extra: value.clone(),
    })
}

/// Emit canonical messages for one chat_message node, including a leading
/// `author: "reasoning"` message when a thinking block is present.
fn push_chat_message(
    out: &mut Vec<CanonicalMessage>,
    value: &serde_json::Value,
    node_created_at: i64,
) {
    let thinking_text = value
        .get("thinking")
        .and_then(|t| t.get("thinking"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    if !thinking_text.is_empty() {
        let timestamp = value
            .pointer("/metadata/created_at")
            .and_then(parse_timestamp)
            .or_else(|| parse_timestamp(&serde_json::json!(node_created_at)));
        out.push(CanonicalMessage {
            idx: 0,
            role: MessageRole::Assistant,
            content: thinking_text,
            timestamp,
            author: Some("reasoning".to_string()),
            tool_calls: vec![],
            tool_results: vec![],
            extra: serde_json::Value::Null,
        });
    }

    if let Some(msg) = chat_message_to_canonical(value, node_created_at) {
        out.push(msg);
    }
}

// ---------------------------------------------------------------------------
// ATIF transcript (.json) → CanonicalSession
// ---------------------------------------------------------------------------

/// Read an ATIF transcript export (`transcripts/<id>.json`).
fn read_atif_transcript(path: &Path) -> anyhow::Result<CanonicalSession> {
    debug!(path = %path.display(), "reading Devin ATIF transcript");

    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read transcript {}", path.display()))?;
    let doc: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("{} is not valid JSON", path.display()))?;

    let schema_version = doc
        .get("schema_version")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if !schema_version.starts_with("ATIF") {
        anyhow::bail!(
            "{} does not look like a Devin transcript (schema_version={schema_version:?}); \
             expected an ATIF-v* export",
            path.display()
        );
    }

    let session_id = doc
        .get("session_id")
        .and_then(|v| v.as_str())
        .map(ToString::to_string)
        .or_else(|| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .map(ToString::to_string)
        })
        .unwrap_or_else(|| "unknown".to_string());

    let model_name = doc
        .pointer("/agent/model_name")
        .and_then(|v| v.as_str())
        .map(ToString::to_string);

    let mut messages: Vec<CanonicalMessage> = Vec::new();
    let mut started_at: Option<i64> = None;
    let mut ended_at: Option<i64> = None;

    if let Some(steps) = doc.get("steps").and_then(|v| v.as_array()) {
        for step in steps {
            let Some(msg) = atif_step_to_messages(step) else {
                continue;
            };
            for m in msg {
                if let Some(ts) = m.timestamp {
                    started_at = Some(started_at.map_or(ts, |s: i64| s.min(ts)));
                    ended_at = Some(ended_at.map_or(ts, |e: i64| e.max(ts)));
                }
                messages.push(m);
            }
        }
    }

    reindex_messages(&mut messages);

    let title = messages
        .iter()
        .find(|m| m.role == MessageRole::User)
        .map(|m| truncate_title(&m.content, 100));

    let mut metadata = serde_json::Map::new();
    metadata.insert(
        "source".into(),
        serde_json::Value::String("devin".to_string()),
    );
    metadata.insert(
        "format".into(),
        serde_json::Value::String("atif-transcript".to_string()),
    );
    metadata.insert(
        "schema_version".into(),
        serde_json::Value::String(schema_version.to_string()),
    );
    if let Some(agent) = doc.get("agent") {
        metadata.insert("agent".into(), agent.clone());
    }
    if let Some(native) = doc.get("session_id").and_then(|v| v.as_str()) {
        metadata.insert(
            crate::model::NATIVE_NAME_META_KEY.into(),
            serde_json::Value::String(native.to_string()),
        );
    }

    debug!(
        session_id,
        messages = messages.len(),
        "Devin ATIF transcript parsed"
    );

    Ok(CanonicalSession {
        session_id,
        provider_slug: "devin".to_string(),
        workspace: None,
        title,
        started_at,
        ended_at,
        messages,
        metadata: serde_json::Value::Object(metadata),
        source_path: path.to_path_buf(),
        model_name,
    })
}

/// Convert one ATIF step into canonical message(s) — a leading reasoning
/// message when `reasoning_content` is present, then the step itself.
fn atif_step_to_messages(step: &serde_json::Value) -> Option<Vec<CanonicalMessage>> {
    let source = step.get("source").and_then(|v| v.as_str()).unwrap_or("");
    let role = normalize_role(source);

    let timestamp = step.get("timestamp").and_then(parse_timestamp);

    // Tool calls: {tool_call_id, function_name, arguments}.
    let mut tool_calls = Vec::new();
    if let Some(arr) = step.get("tool_calls").and_then(|v| v.as_array()) {
        for call in arr {
            let Some(obj) = call.as_object() else {
                continue;
            };
            tool_calls.push(ToolCall {
                id: obj
                    .get("tool_call_id")
                    .and_then(|v| v.as_str())
                    .map(ToString::to_string),
                name: obj
                    .get("function_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                arguments: obj
                    .get("arguments")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null),
            });
        }
    }

    // Observation results ride on the same agent step:
    // observation.results[] = {source_call_id, content}.
    let mut tool_results = Vec::new();
    if let Some(results) = step
        .pointer("/observation/results")
        .and_then(|v| v.as_array())
    {
        for res in results {
            let Some(obj) = res.as_object() else { continue };
            tool_results.push(ToolResult {
                call_id: obj
                    .get("source_call_id")
                    .and_then(|v| v.as_str())
                    .map(ToString::to_string),
                content: obj
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                is_error: false,
            });
        }
    }

    let mut content = step
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let reasoning = step
        .get("reasoning_content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    if content.trim().is_empty() && tool_calls.is_empty() && tool_results.is_empty() {
        content = reasoning.clone();
    }

    if content.trim().is_empty() && tool_calls.is_empty() && tool_results.is_empty() {
        return None;
    }

    let mut out = Vec::with_capacity(2);
    if !reasoning.is_empty() {
        out.push(CanonicalMessage {
            idx: 0,
            role: MessageRole::Assistant,
            content: reasoning,
            timestamp,
            author: Some("reasoning".to_string()),
            tool_calls: vec![],
            tool_results: vec![],
            extra: serde_json::Value::Null,
        });
    }

    out.push(CanonicalMessage {
        idx: 0,
        role,
        content,
        timestamp,
        author: step
            .get("model_name")
            .and_then(|v| v.as_str())
            .map(ToString::to_string),
        tool_calls,
        tool_results,
        extra: step.clone(),
    });

    Some(out)
}

// ---------------------------------------------------------------------------
// Provider impl
// ---------------------------------------------------------------------------

impl Provider for Devin {
    fn name(&self) -> &str {
        "Devin CLI"
    }

    fn slug(&self) -> &str {
        "devin"
    }

    fn cli_alias(&self) -> &str {
        "dvn"
    }

    fn detect(&self) -> DetectionResult {
        let mut evidence = Vec::new();
        let mut installed = false;

        if which::which("devin").is_ok() {
            evidence.push("devin binary found in PATH".to_string());
            installed = true;
        }

        let dbs = Self::db_candidates();
        if !dbs.is_empty() {
            evidence.push(format!("found {} sessions.db database(s)", dbs.len()));
            installed = true;
        }

        let cli_dirs: Vec<PathBuf> = Self::cli_dir_candidates()
            .into_iter()
            .filter(|d| d.is_dir())
            .collect();
        if !cli_dirs.is_empty() && dbs.is_empty() {
            evidence.push(format!(
                "Devin CLI data dir exists: {}",
                cli_dirs[0].display()
            ));
            installed = true;
        }

        trace!(provider = "devin", ?evidence, installed, "detection");
        DetectionResult {
            installed,
            version: None,
            evidence,
        }
    }

    fn session_roots(&self) -> Vec<PathBuf> {
        let mut roots = Self::db_candidates();
        roots.extend(Self::transcript_dirs());
        roots
    }

    fn list_sessions(&self) -> Option<Vec<(String, PathBuf)>> {
        let mut results = Vec::new();
        for db_path in Self::db_candidates() {
            results.extend(Self::list_sessions_in_db(&db_path));
        }
        Some(results)
    }

    fn owns_session(&self, session_id: &str) -> Option<PathBuf> {
        for db_path in Self::db_candidates() {
            let Ok(conn) = Self::open_db(&db_path) else {
                continue;
            };
            if !Self::table_exists(&conn, SESSIONS_TABLE) {
                continue;
            }

            // Exact match first, then case-insensitive (ids are lowercase
            // slugs, but be robust to user-typed mixed case).
            let found_id: Option<String> = conn
                .query_row(
                    "SELECT id FROM sessions WHERE id = ?1",
                    rusqlite::params![session_id],
                    |row| row.get(0),
                )
                .ok()
                .or_else(|| {
                    conn.query_row(
                        "SELECT id FROM sessions WHERE id = ?1 COLLATE NOCASE",
                        rusqlite::params![session_id],
                        |row| row.get(0),
                    )
                    .ok()
                });

            if let Some(id) = found_id {
                let virtual_path = Self::virtual_session_path(&db_path, &id);
                debug!(
                    db = %db_path.display(),
                    session = %virtual_path.display(),
                    session_id = %id,
                    "found Devin session"
                );
                return Some(virtual_path);
            }
        }
        None
    }

    fn read_session(&self, path: &Path) -> anyhow::Result<CanonicalSession> {
        debug!(path = %path.display(), "reading Devin session");

        // Virtual path (`.../sessions.db/<encoded-session-id>`) from discovery.
        if let Some((db_path, session_id)) = Self::parse_virtual_path(path) {
            return Self::read_session_from_db(&db_path, &session_id);
        }

        // ATIF transcript export (`transcripts/<id>.json`).
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            if Self::looks_like_atif(path) {
                return read_atif_transcript(path);
            }
            anyhow::bail!(
                "{} is not a Devin ATIF transcript (expected schema_version ATIF-v*)",
                path.display()
            );
        }

        // Bare sessions.db — read the most recently active session.
        if path.is_file()
            && (path.file_name().and_then(|n| n.to_str()) == Some(DB_FILENAME)
                || matches!(
                    path.extension().and_then(|e| e.to_str()),
                    Some("db") | Some("sqlite") | Some("sqlite3")
                ))
        {
            let conn = Self::open_db(path)?;
            if !Self::table_exists(&conn, SESSIONS_TABLE) {
                anyhow::bail!("{} has no Devin `sessions` table", path.display());
            }
            let latest: Option<String> = conn
                .query_row(
                    "SELECT id FROM sessions ORDER BY last_activity_at DESC LIMIT 1",
                    [],
                    |row| row.get(0),
                )
                .ok();
            let Some(session_id) = latest else {
                anyhow::bail!("no Devin sessions found in {}", path.display());
            };
            return Self::read_session_from_db(path, &session_id);
        }

        anyhow::bail!(
            "unsupported Devin session path: {} (expected a sessions.db virtual path \
             or an ATIF transcript .json)",
            path.display()
        )
    }

    fn write_session(
        &self,
        _session: &CanonicalSession,
        _opts: &WriteOptions,
    ) -> anyhow::Result<WrittenSession> {
        // Devin sessions live in a live SQLite WAL database whose resumable
        // state is a forest of `message_nodes` chains plus auxiliary tables
        // (`tool_call_state`, `rendered_commits`, `subagent_heads`,
        // `message_search` FTS) owned and versioned by the Devin CLI. There
        // is no documented import path — transcripts are export-only — and
        // direct row insertion cannot be verified safe against the user's
        // real session history, so we refuse rather than write an
        // un-resumable stub.
        Err(anyhow::anyhow!(
            "Devin CLI is read/resume-only: casr cannot create a resumable \
             Devin session from another provider's history (sessions.db is a \
             CLI-owned forest store with no supported import path). Use Devin \
             as a conversion SOURCE (e.g. `casr cc resume <devin-session-id> \
             --source dvn`), not a target."
        ))
    }

    fn target_capable(&self) -> bool {
        false
    }

    fn resume_command(&self, session_id: &str) -> String {
        format!(
            "devin --resume {}",
            crate::providers::shell_quote_arg(session_id)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::Provider;
    use rusqlite::Connection;
    use serde_json::json;

    // -----------------------------------------------------------------------
    // Helpers — build a minimal Devin sessions.db in a temp dir.
    // -----------------------------------------------------------------------

    fn create_db(dir: &Path) -> PathBuf {
        let db_path = dir.join(DB_FILENAME);
        let conn = Connection::open(&db_path).expect("create test db");
        conn.execute_batch(
            "CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                working_directory TEXT NOT NULL,
                backend_type TEXT NOT NULL,
                model TEXT NOT NULL,
                agent_mode TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                last_activity_at INTEGER NOT NULL,
                title TEXT,
                main_chain_id INTEGER,
                shell_last_seen_index INTEGER DEFAULT 0,
                cogs_json TEXT,
                workspace_dirs TEXT,
                hidden INTEGER NOT NULL DEFAULT 0,
                metadata TEXT
            );
            CREATE TABLE message_nodes (
                row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                node_id INTEGER NOT NULL,
                parent_node_id INTEGER,
                chat_message TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                metadata TEXT,
                UNIQUE(session_id, node_id)
            );",
        )
        .expect("create schema");
        conn.close().expect("close");
        db_path
    }

    fn insert_session(conn: &Connection, id: &str, main_chain_id: i64, hidden: i64) {
        conn.execute(
            "INSERT INTO sessions (id, working_directory, backend_type, model, agent_mode,
                                   created_at, last_activity_at, title, main_chain_id, hidden)
             VALUES (?1, '/tmp/work', 'windsurf', 'swe-2-flash-medium', 'bypass',
                     1700000000, 1700000100, 'Test title', ?2, ?3)",
            rusqlite::params![id, main_chain_id, hidden],
        )
        .expect("insert session");
    }

    fn insert_node(
        conn: &Connection,
        session_id: &str,
        node_id: i64,
        parent: Option<i64>,
        chat: &serde_json::Value,
    ) {
        conn.execute(
            "INSERT INTO message_nodes (session_id, node_id, parent_node_id, chat_message, created_at)
             VALUES (?1, ?2, ?3, ?4, 1700000000)",
            rusqlite::params![session_id, node_id, parent, chat.to_string()],
        )
        .expect("insert node");
    }

    fn user_msg(text: &str) -> serde_json::Value {
        json!({
            "message_id": uuid::Uuid::new_v4().to_string(),
            "role": "user",
            "content": text,
            "metadata": {"created_at": "2026-01-01T00:00:00Z", "is_user_input": true}
        })
    }

    fn assistant_msg(text: &str) -> serde_json::Value {
        json!({
            "message_id": uuid::Uuid::new_v4().to_string(),
            "role": "assistant",
            "content": text,
            "metadata": {"created_at": "2026-01-01T00:00:01Z"}
        })
    }

    /// Build a db with one session whose main chain is 0→1→2, plus an
    /// abandoned branch 1→3 (node 3 must NOT appear in the output).
    fn seeded_db(dir: &Path) -> PathBuf {
        let db_path = create_db(dir);
        let conn = Connection::open(&db_path).expect("open test db");
        insert_session(&conn, "test-session", 2, 0);
        insert_node(&conn, "test-session", 0, None, &user_msg("hello devin"));
        insert_node(
            &conn,
            "test-session",
            1,
            Some(0),
            &assistant_msg("hi there"),
        );
        insert_node(&conn, "test-session", 2, Some(1), &assistant_msg("done"));
        // Abandoned sibling branch off node 1 — excluded from main chain.
        insert_node(
            &conn,
            "test-session",
            3,
            Some(1),
            &user_msg("abandoned branch"),
        );
        conn.close().expect("close");
        db_path
    }

    // -----------------------------------------------------------------------
    // Virtual path round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn virtual_path_round_trip() {
        let tmp = tempfile::tempdir().expect("tmp");
        let db = tmp.path().join(DB_FILENAME);
        std::fs::write(&db, b"").expect("touch db");
        let vp = Devin::virtual_session_path(&db, "spiral-pump");
        assert_eq!(
            vp,
            db.join("spiral-pump"),
            "virtual path joins encoded id onto db path"
        );
        let (parsed_db, id) = Devin::parse_virtual_path(&vp).expect("parse");
        assert_eq!(parsed_db, db);
        assert_eq!(id, "spiral-pump");
    }

    #[test]
    fn parse_virtual_path_rejects_non_sessions_db() {
        let tmp = tempfile::tempdir().expect("tmp");
        // A real file named opencode.db must not be claimed by Devin.
        let oc = tmp.path().join("opencode.db");
        std::fs::write(&oc, b"").expect("touch");
        assert!(Devin::parse_virtual_path(&oc.join("ses_123")).is_none());
        // The bare db path itself is not a virtual session path.
        let db = tmp.path().join(DB_FILENAME);
        std::fs::write(&db, b"").expect("touch");
        assert!(Devin::parse_virtual_path(&db).is_none());
    }

    #[test]
    fn parse_virtual_path_decodes_urlencoding() {
        let tmp = tempfile::tempdir().expect("tmp");
        let db = tmp.path().join(DB_FILENAME);
        std::fs::write(&db, b"").expect("touch db");
        let vp = Devin::virtual_session_path(&db, "a b/c");
        let (_, id) = Devin::parse_virtual_path(&vp).expect("parse");
        assert_eq!(id, "a b/c");
    }

    // -----------------------------------------------------------------------
    // chat_message → CanonicalMessage
    // -----------------------------------------------------------------------

    #[test]
    fn chat_message_user_role() {
        let v = user_msg("fix the bug");
        let msg = chat_message_to_canonical(&v, 1700000000).expect("message");
        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.content, "fix the bug");
        assert_eq!(msg.timestamp, Some(1767225600000)); // 2026-01-01T00:00:00Z
    }

    #[test]
    fn chat_message_assistant_with_tool_calls() {
        let v = json!({
            "message_id": "m1",
            "role": "assistant",
            "content": "running tools",
            "tool_calls": [{
                "id": "call_1",
                "name": "exec",
                "arguments": {"command": "ls"},
                "index": 0,
                "kind": "function"
            }],
            "thinking": {"thinking": "let me think", "signature": ""},
            "metadata": {"created_at": "2026-01-01T00:00:02Z"}
        });
        let msgs = {
            let mut out = Vec::new();
            push_chat_message(&mut out, &v, 1700000000);
            out
        };
        // reasoning message + assistant message
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, MessageRole::Assistant);
        assert_eq!(msgs[0].author.as_deref(), Some("reasoning"));
        assert_eq!(msgs[0].content, "let me think");
        assert_eq!(msgs[1].role, MessageRole::Assistant);
        assert_eq!(msgs[1].tool_calls.len(), 1);
        assert_eq!(msgs[1].tool_calls[0].name, "exec");
        assert_eq!(msgs[1].tool_calls[0].id.as_deref(), Some("call_1"));
    }

    #[test]
    fn chat_message_tool_result_is_error() {
        let v = json!({
            "message_id": "m2",
            "role": "tool",
            "content": "boom",
            "tool_call_id": "call_1",
            "metadata": {
                "created_at": "2026-01-01T00:00:03Z",
                "extensions": {"chisel/tool_result_meta": {"success": false, "kind": "execute"}}
            }
        });
        let msg = chat_message_to_canonical(&v, 1700000000).expect("message");
        assert_eq!(msg.role, MessageRole::Tool);
        assert_eq!(msg.content, "boom");
        assert_eq!(msg.tool_results.len(), 1);
        assert_eq!(msg.tool_results[0].call_id.as_deref(), Some("call_1"));
        assert!(msg.tool_results[0].is_error);
    }

    #[test]
    fn chat_message_empty_is_skipped() {
        let v = json!({"role": "assistant", "content": ""});
        assert!(chat_message_to_canonical(&v, 0).is_none());
    }

    #[test]
    fn chat_message_tool_only_kept() {
        let v = json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c", "name": "read", "arguments": {}}]
        });
        let msg = chat_message_to_canonical(&v, 0).expect("kept");
        assert_eq!(msg.tool_calls.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Chain reconstruction
    // -----------------------------------------------------------------------

    #[test]
    fn main_chain_walks_parents_and_ignores_branches() {
        let tmp = tempfile::tempdir().expect("tmp");
        let db_path = seeded_db(tmp.path());
        let conn = Devin::open_db(&db_path).expect("open");
        let chain = main_chain_node_ids(&conn, "test-session", 2).expect("chain");
        assert_eq!(chain, vec![0, 1, 2]);
    }

    #[test]
    fn main_chain_cycle_is_bounded() {
        let tmp = tempfile::tempdir().expect("tmp");
        let db_path = create_db(tmp.path());
        let conn = Connection::open(&db_path).expect("open");
        insert_session(&conn, "cyc", 1, 0);
        insert_node(&conn, "cyc", 0, Some(1), &user_msg("a"));
        insert_node(&conn, "cyc", 1, Some(0), &assistant_msg("b"));
        conn.close().expect("close");

        let conn = Devin::open_db(&db_path).expect("reopen");
        let chain = main_chain_node_ids(&conn, "cyc", 1).expect("chain");
        // Cycle 1→0→1… stops after visiting each node once.
        assert!(chain.len() <= 2);
        assert!(chain.contains(&0) && chain.contains(&1));
    }

    // -----------------------------------------------------------------------
    // End-to-end DB read
    // -----------------------------------------------------------------------

    #[test]
    fn read_session_from_db_parses_main_chain() {
        let tmp = tempfile::tempdir().expect("tmp");
        let db_path = seeded_db(tmp.path());

        let session = Devin::read_session_from_db(&db_path, "test-session").expect("read session");

        assert_eq!(session.session_id, "test-session");
        assert_eq!(session.provider_slug, "devin");
        assert_eq!(session.title.as_deref(), Some("Test title"));
        assert_eq!(
            session.workspace.as_ref().map(|p| p.to_str().unwrap()),
            Some("/tmp/work")
        );
        assert_eq!(session.model_name.as_deref(), Some("swe-2-flash-medium"));
        assert_eq!(session.started_at, Some(1700000000000));
        assert_eq!(session.ended_at, Some(1700000100000));

        // Main chain: user + 2 assistant (the abandoned branch is excluded).
        let contents: Vec<&str> = session
            .messages
            .iter()
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(contents, vec!["hello devin", "hi there", "done"]);
        assert_eq!(session.messages[0].role, MessageRole::User);
        assert_eq!(session.messages[1].role, MessageRole::Assistant);
        assert_eq!(session.messages[2].idx, 2);
    }

    #[test]
    fn read_session_missing_id_errors() {
        let tmp = tempfile::tempdir().expect("tmp");
        let db_path = seeded_db(tmp.path());
        let err = Devin::read_session_from_db(&db_path, "nope").expect_err("should fail");
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn read_session_rejects_non_devin_db() {
        let tmp = tempfile::tempdir().expect("tmp");
        // A sqlite db without the Devin tables.
        let db_path = tmp.path().join(DB_FILENAME);
        let conn = Connection::open(&db_path).expect("open");
        conn.execute_batch("CREATE TABLE something_else (id TEXT);")
            .expect("schema");
        conn.close().expect("close");
        let err = Devin::read_session_from_db(&db_path, "x").expect_err("should fail");
        assert!(err.to_string().contains("sessions.db"));
    }

    #[test]
    fn read_session_via_virtual_path() {
        let tmp = tempfile::tempdir().expect("tmp");
        let db_path = seeded_db(tmp.path());
        let provider = Devin;
        let vp = Devin::virtual_session_path(&db_path, "test-session");
        let session = provider.read_session(&vp).expect("read via virtual path");
        assert_eq!(session.session_id, "test-session");
    }

    #[test]
    fn read_session_via_bare_db_path_picks_latest() {
        let tmp = tempfile::tempdir().expect("tmp");
        let db_path = seeded_db(tmp.path());
        let provider = Devin;
        let session = provider.read_session(&db_path).expect("read bare db");
        assert_eq!(session.session_id, "test-session");
    }

    #[test]
    fn list_sessions_in_db_returns_visible_only() {
        let tmp = tempfile::tempdir().expect("tmp");
        let db_path = seeded_db(tmp.path());
        let conn = Connection::open(&db_path).expect("open");
        insert_session(&conn, "hidden-one", 0, 1);
        conn.close().expect("close");

        let listed = Devin::list_sessions_in_db(&db_path);
        let ids: Vec<String> = listed.iter().map(|(id, _)| id.clone()).collect();
        assert_eq!(ids, vec!["test-session".to_string()]);
        assert!(listed[0].1.ends_with("test-session"));
    }

    #[test]
    fn owns_session_finds_id() {
        let tmp = tempfile::tempdir().expect("tmp");
        let db_path = seeded_db(tmp.path());
        TEST_DB_PATH_OVERRIDE.with(|cell| *cell.borrow_mut() = Some(db_path.clone()));

        let provider = Devin;
        let found = provider.owns_session("test-session").expect("found");
        assert!(found.ends_with("test-session"));
        assert!(Devin::parse_virtual_path(&found).is_some());
        assert!(provider.owns_session("missing-id").is_none());

        TEST_DB_PATH_OVERRIDE.with(|cell| *cell.borrow_mut() = None);
    }

    #[test]
    fn list_sessions_uses_override() {
        let tmp = tempfile::tempdir().expect("tmp");
        let db_path = seeded_db(tmp.path());
        TEST_DB_PATH_OVERRIDE.with(|cell| *cell.borrow_mut() = Some(db_path));

        let listed = Devin.list_sessions().expect("some");
        let ids: Vec<String> = listed.iter().map(|(id, _)| id.clone()).collect();
        assert_eq!(ids, vec!["test-session".to_string()]);

        TEST_DB_PATH_OVERRIDE.with(|cell| *cell.borrow_mut() = None);
    }

    // -----------------------------------------------------------------------
    // ATIF transcript
    // -----------------------------------------------------------------------

    #[test]
    fn atif_step_maps_roles_tools_and_observation() {
        let step = json!({
            "step_id": 7,
            "timestamp": "2026-11-15T17:31:27.893385458+00:00",
            "source": "agent",
            "message": "checking the PR",
            "model_name": "swe-2-flash-medium",
            "tool_calls": [{
                "tool_call_id": "call_abc",
                "function_name": "exec",
                "arguments": {"command": "gh pr view 46"}
            }],
            "observation": {
                "results": [{"source_call_id": "call_abc", "content": "pr output"}]
            },
            "reasoning_content": "need to check CI"
        });
        let msgs = atif_step_to_messages(&step).expect("messages");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].author.as_deref(), Some("reasoning"));
        assert_eq!(msgs[0].content, "need to check CI");
        assert_eq!(msgs[1].role, MessageRole::Assistant);
        assert_eq!(msgs[1].tool_calls[0].name, "exec");
        assert_eq!(msgs[1].tool_calls[0].id.as_deref(), Some("call_abc"));
        assert_eq!(msgs[1].tool_results.len(), 1);
        assert_eq!(msgs[1].tool_results[0].call_id.as_deref(), Some("call_abc"));
        assert_eq!(msgs[1].tool_results[0].content, "pr output");
        assert_eq!(msgs[1].author.as_deref(), Some("swe-2-flash-medium"));
    }

    #[test]
    fn atif_transcript_end_to_end() {
        let tmp = tempfile::tempdir().expect("tmp");
        let path = tmp.path().join("demo-session.json");
        let doc = json!({
            "schema_version": "ATIF-v1.7",
            "session_id": "demo-session",
            "agent": {"name": "devin", "version": "3000.12.57", "model_name": "SWE-2 Flash Medium"},
            "steps": [
                {"step_id": 1, "timestamp": "2026-11-15T16:00:00Z", "source": "system", "message": "You are Devin."},
                {"step_id": 2, "timestamp": "2026-11-15T16:01:00Z", "source": "user", "message": "hello"},
                {"step_id": 3, "timestamp": "2026-11-15T16:02:00Z", "source": "agent", "message": "hi!"}
            ],
            "final_metrics": {}
        });
        std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap()).unwrap();

        let provider = Devin;
        let session = provider.read_session(&path).expect("read transcript");
        assert_eq!(session.session_id, "demo-session");
        assert_eq!(session.model_name.as_deref(), Some("SWE-2 Flash Medium"));
        assert_eq!(session.messages.len(), 3);
        assert_eq!(session.messages[1].role, MessageRole::User);
        assert_eq!(session.messages[2].content, "hi!");
        assert_eq!(session.title.as_deref(), Some("hello"));
    }

    #[test]
    fn atif_rejects_non_atif_json() {
        let tmp = tempfile::tempdir().expect("tmp");
        let path = tmp.path().join("other.json");
        std::fs::write(&path, r#"{"schema_version":"SomethingElse","steps":[]}"#).unwrap();
        // looks_like_atif requires the literal "ATIF" marker in the prefix.
        assert!(!Devin::looks_like_atif(&path));
        let provider = Devin;
        assert!(provider.read_session(&path).is_err());
    }

    #[test]
    fn looks_like_atif_detects_marker() {
        let tmp = tempfile::tempdir().expect("tmp");
        let path = tmp.path().join("t.json");
        std::fs::write(&path, r#"{"schema_version":"ATIF-v1.7","session_id":"x"}"#).unwrap();
        assert!(Devin::looks_like_atif(&path));
    }

    // -----------------------------------------------------------------------
    // Provider surface
    // -----------------------------------------------------------------------

    #[test]
    fn provider_metadata_and_resume() {
        let provider = Devin;
        assert_eq!(provider.name(), "Devin CLI");
        assert_eq!(provider.slug(), "devin");
        assert_eq!(provider.cli_alias(), "dvn");
        assert!(!provider.target_capable());
        assert_eq!(
            provider.resume_command("spiral-pump"),
            "devin --resume spiral-pump"
        );
        assert_eq!(provider.resume_command("odd id"), "devin --resume 'odd id'");
    }

    #[test]
    fn write_session_refuses() {
        let provider = Devin;
        let session = CanonicalSession {
            session_id: "x".to_string(),
            provider_slug: "test".to_string(),
            workspace: None,
            title: None,
            started_at: None,
            ended_at: None,
            messages: vec![],
            metadata: json!({}),
            source_path: PathBuf::from("/tmp/x"),
            model_name: None,
        };
        let err = provider
            .write_session(&session, &WriteOptions::default())
            .expect_err("write must fail");
        assert!(err.to_string().contains("read/resume-only"));
    }
}

//! Hermes provider — reads/writes sessions from Hermes Agent's SQLite database.
//!
//! Hermes Agent stores sessions in a SQLite database at `~/.hermes/state.db`
//! (or `$HERMES_HOME/state.db`). The database has `sessions` (metadata) and
//! `messages` (conversation turns) tables.
//!
//! Since all sessions live in one DB file, we use a virtual path convention:
//! `<db_path>#<session_id>` to identify individual sessions. Virtual paths
//! are returned by [`list_sessions`] and [`owns_session`]; [`read_session`]
//! parses the virtual path to extract the DB path and session ID.
//!
//! ## Environment overrides
//!
//! - `HERMES_HOME` — override the data directory (default `~/.hermes`).
//!
//! ## Resume command
//!
//! Hermes CLI supports `--resume <session-id>` to resume a session.

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use anyhow::Context;
use tracing::{debug, info, trace};

use crate::discovery::DetectionResult;
use crate::model::{
    CanonicalMessage, CanonicalSession, MessageRole, ToolCall, ToolResult, normalize_role,
    reindex_messages, truncate_title,
};
use crate::providers::{Provider, WriteOptions, WrittenSession};

/// Separator used to create virtual session paths: `<db_path>#<session_id>`.
const VIRTUAL_SEPARATOR: char = '#';

/// Suffix appended to IDs written by CASR so Hermes users can identify
/// converted sessions.
const CASR_ID_SUFFIX: &str = "-casr";

// Thread-local override for the Hermes home directory (used in tests).
// Rust 2024 marks `set_var`/`remove_var` as unsafe and the crate forbids
// unsafe code, so we provide a safe test override via thread-local storage.
thread_local! {
    static TEST_HERMES_HOME: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

/// Hermes provider implementation.
pub struct Hermes;

impl Hermes {
    /// Resolve the Hermes data directory.
    ///
    /// Resolution precedence:
    /// 1. Test override (tests only — safe env-var alternative)
    /// 2. `HERMES_HOME` env var
    /// 3. `~/.hermes` (POSIX) / `%LOCALAPPDATA%/hermes` (Windows)
    fn hermes_home() -> Option<PathBuf> {
        // Check test override first (avoids unsafe `set_var` on Rust 2024).
        let test_override = TEST_HERMES_HOME.with(|cell| cell.borrow().clone());
        if let Some(home) = test_override {
            return Some(home);
        }
        Self::hermes_home_impl(std::env::var("HERMES_HOME").ok())
    }

    /// Inner implementation factored out for testability without env-var
    /// manipulation (which is `unsafe` on Rust 2024 nightly).
    fn hermes_home_impl(hermes_home_env: Option<String>) -> Option<PathBuf> {
        if let Some(ref home) = hermes_home_env {
            return Some(PathBuf::from(home));
        }
        #[cfg(target_os = "windows")]
        {
            if let Some(local) = std::env::var_os("LOCALAPPDATA") {
                let p = PathBuf::from(local).join("hermes");
                if p.join("state.db").is_file() || p.is_dir() {
                    return Some(p);
                }
            }
        }
        dirs::home_dir().map(|h| h.join(".hermes"))
    }

    /// Path to the SQLite state database (production — reads env vars).
    fn db_path() -> Option<PathBuf> {
        let home = Self::hermes_home()?;
        Some(home.join("state.db"))
    }

    /// Parse a virtual path like `<db_path>#<session_id>` into its components.
    /// Returns `(db_path, Some(session_id))`. When no separator is found
    /// (bare DB path), returns `(path, None)`.
    fn parse_virtual_path(path: &Path) -> (PathBuf, Option<String>) {
        let path_str = path.to_string_lossy();
        if let Some(pos) = path_str.rfind(VIRTUAL_SEPARATOR) {
            let db_path = PathBuf::from(&path_str[..pos]);
            let session_id = path_str[pos + 1..].to_string();
            (db_path, Some(session_id))
        } else {
            (path.to_path_buf(), None)
        }
    }

    /// Build a virtual path from a DB path and session ID.
    fn build_virtual_path(db_path: &Path, session_id: &str) -> PathBuf {
        let virt = format!("{}{}{}", db_path.display(), VIRTUAL_SEPARATOR, session_id);
        PathBuf::from(virt)
    }

    /// Open a connection to the Hermes state database.
    fn open_db(db_path: &Path) -> anyhow::Result<rusqlite::Connection> {
        let conn = rusqlite::Connection::open(db_path).with_context(|| {
            format!(
                "failed to open Hermes state database: {}",
                db_path.display()
            )
        })?;
        Ok(conn)
    }

    /// Read a single session from the Hermes SQLite database.
    fn read_session_from_db(db_path: &Path, session_id: &str) -> anyhow::Result<CanonicalSession> {
        debug!(db = %db_path.display(), session_id, "reading Hermes session from DB");
        let conn = Self::open_db(db_path)?;

        // Read session metadata.
        let session_row = conn
            .query_row(
                "SELECT id, source, model, cwd, title, started_at, ended_at,
                        message_count, input_tokens, output_tokens, reasoning_tokens
                 FROM sessions WHERE id = ?1",
                rusqlite::params![session_id],
                |row| {
                    let id: String = row.get("id")?;
                    let source: Option<String> = row.get("source")?;
                    let model: Option<String> = row.get("model")?;
                    let cwd: Option<String> = row.get("cwd")?;
                    let title: Option<String> = row.get("title")?;
                    let started_at: Option<f64> = row.get("started_at")?;
                    let ended_at: Option<f64> = row.get("ended_at")?;
                    let message_count: Option<i64> = row.get("message_count")?;
                    let input_tokens: Option<i64> = row.get("input_tokens")?;
                    let output_tokens: Option<i64> = row.get("output_tokens")?;
                    let reasoning_tokens: Option<i64> = row.get("reasoning_tokens")?;
                    Ok((
                        id,
                        source,
                        model,
                        cwd,
                        title,
                        started_at,
                        ended_at,
                        message_count,
                        input_tokens,
                        output_tokens,
                        reasoning_tokens,
                    ))
                },
            )
            .with_context(|| format!("session not found in Hermes DB: {session_id}"))?;

        let (
            id,
            _source,
            model,
            cwd,
            title,
            started_at_f,
            ended_at_f,
            _message_count,
            input_tokens,
            output_tokens,
            reasoning_tokens,
        ) = session_row;

        // Read messages (active only — soft-deleted / rewound messages are excluded).
        let mut stmt = conn
            .prepare(
                "SELECT id, role, content, tool_call_id, tool_calls, tool_name,
                        timestamp, reasoning, reasoning_content
                 FROM messages
                 WHERE session_id = ?1 AND active = 1
                 ORDER BY id ASC",
            )
            .with_context(|| "failed to prepare messages query")?;

        let msg_rows = stmt
            .query_map(rusqlite::params![session_id], |row| {
                let msg_id: i64 = row.get("id")?;
                let role: String = row.get("role")?;
                let content: Option<String> = row.get("content")?;
                let tool_call_id: Option<String> = row.get("tool_call_id")?;
                let tool_calls_str: Option<String> = row.get("tool_calls")?;
                let tool_name: Option<String> = row.get("tool_name")?;
                let timestamp: Option<f64> = row.get("timestamp")?;
                let reasoning: Option<String> = row.get("reasoning")?;
                let reasoning_content: Option<String> = row.get("reasoning_content")?;
                Ok((
                    msg_id,
                    role,
                    content,
                    tool_call_id,
                    tool_calls_str,
                    tool_name,
                    timestamp,
                    reasoning,
                    reasoning_content,
                ))
            })
            .with_context(|| "failed to query messages")?;

        let mut messages: Vec<CanonicalMessage> = Vec::new();
        for msg_row in msg_rows {
            let (
                _msg_id,
                role_str,
                content,
                tool_call_id,
                tool_calls_str,
                tool_name,
                timestamp_f,
                reasoning,
                reasoning_content,
            ) = msg_row?;

            let role = normalize_role(&role_str);

            // Parse tool_calls JSON stored in the DB column.
            let tool_calls: Vec<ToolCall> = if let Some(ref tc_str) = tool_calls_str {
                serde_json::from_str(tc_str).unwrap_or_default()
            } else {
                vec![]
            };

            // Tool messages: produce a ToolResult containing the output.
            // Hermes stores tool outputs as messages with role="tool",
            // optionally with a tool_call_id pointing back to the originating call.
            let tool_call_id_for_tool = tool_call_id.clone();
            let tool_results: Vec<ToolResult> = if role == MessageRole::Tool {
                let call_id = tool_call_id_for_tool
                    .or_else(|| tool_name.clone())
                    .filter(|s| !s.is_empty());
                vec![ToolResult {
                    call_id,
                    content: content.clone().unwrap_or_default(),
                    is_error: false,
                }]
            } else {
                vec![]
            };

            // Convert SQLite REAL timestamp (seconds) to epoch milliseconds.
            let ts_ms = timestamp_f.map(|f| (f * 1000.0) as i64);

            // Combine content with reasoning text to preserve it through
            // cross-provider conversion.
            let effective_content = match (content.as_deref(), reasoning.as_deref()) {
                (Some(c), Some(r)) if !r.is_empty() && !c.trim().is_empty() => {
                    format!("[Reasoning]\n{r}\n\n---\n\n{c}")
                }
                (Some(_c), Some(r)) if !r.is_empty() => format!("[Reasoning]\n{r}"),
                (Some(c), _) => c.to_string(),
                (None, Some(r)) if !r.is_empty() => format!("[Reasoning]\n{r}"),
                _ => String::new(),
            };

            // Set author to the session model for assistant messages.
            let author = match role {
                MessageRole::Assistant => model.clone(),
                _ => None,
            };

            // Reasoning content in messages table marks model-internal thinking.
            // Tag those with author="reasoning" so the pipeline can optionally
            // drop them during cross-provider conversion.
            let effective_author = if reasoning_content
                .as_deref()
                .is_some_and(|rc| !rc.trim().is_empty())
            {
                Some("reasoning".to_string())
            } else {
                author
            };

            // Build extra metadata with Hermes-specific fields.
            let mut extra = serde_json::Map::new();
            if let Some(ts) = timestamp_f {
                extra.insert(
                    "timestamp".to_string(),
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(ts).unwrap_or_else(|| 0.into()),
                    ),
                );
            }
            if let Some(tcid) = &tool_call_id {
                extra.insert(
                    "tool_call_id".to_string(),
                    serde_json::Value::String(tcid.clone()),
                );
            }

            messages.push(CanonicalMessage {
                idx: 0, // reindexed below
                role,
                content: effective_content,
                timestamp: ts_ms,
                author: effective_author,
                tool_calls,
                tool_results,
                extra: serde_json::Value::Object(extra),
            });
        }

        reindex_messages(&mut messages);

        // Build metadata with token usage summary.
        let mut metadata = serde_json::Map::new();
        metadata.insert(
            "source".into(),
            serde_json::Value::String("hermes".to_string()),
        );
        if let Some(it) = input_tokens {
            metadata.insert("input_tokens".into(), serde_json::Value::Number(it.into()));
        }
        if let Some(ot) = output_tokens {
            metadata.insert("output_tokens".into(), serde_json::Value::Number(ot.into()));
        }
        if let Some(rt) = reasoning_tokens {
            metadata.insert(
                "reasoning_tokens".into(),
                serde_json::Value::Number(rt.into()),
            );
        }

        // Convert SQLite REAL timestamps (seconds) to epoch milliseconds.
        let started_at_ms = started_at_f.map(|f| (f * 1000.0) as i64);
        let ended_at_ms = ended_at_f.map(|f| (f * 1000.0) as i64);

        let session_title = title.or_else(|| {
            messages
                .iter()
                .find(|m| m.role == MessageRole::User)
                .map(|m| truncate_title(&m.content, 100))
        });

        let virtual_path = Self::build_virtual_path(db_path, session_id);

        info!(
            session_id,
            messages = messages.len(),
            "Hermes session read from DB"
        );

        Ok(CanonicalSession {
            session_id: id,
            provider_slug: "hermes".to_string(),
            workspace: cwd.map(PathBuf::from),
            title: session_title,
            started_at: started_at_ms,
            ended_at: ended_at_ms,
            messages,
            metadata: serde_json::Value::Object(metadata),
            source_path: virtual_path,
            model_name: model,
        })
    }

    /// Write a session into the Hermes SQLite database.
    ///
    /// Creates the DB and tables if they do not exist, using a minimal schema
    /// compatible with Hermes's native table structure.
    fn write_session_to_db(
        db_path: &Path,
        session: &CanonicalSession,
        target_session_id: &str,
    ) -> anyhow::Result<()> {
        debug!(
            db = %db_path.display(),
            session_id = target_session_id,
            "writing Hermes session to DB"
        );

        // Ensure the parent directory exists so we can create the DB.
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }

        let conn = Self::open_db(db_path)?;

        // Enable WAL mode for concurrent read/write safety.
        conn.execute_batch("PRAGMA journal_mode=WAL")
            .context("failed to set WAL mode")?;

        // Create sessions and messages tables if they do not exist.
        // Uses Hermes-compatible column names so existing Hermes installations
        // can discover converted sessions.
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL DEFAULT 'cli',
                user_id TEXT,
                model TEXT,
                model_config TEXT,
                system_prompt TEXT,
                parent_session_id TEXT,
                started_at REAL NOT NULL DEFAULT ((julianday('now') - 2440587.5) * 86400.0),
                ended_at REAL,
                end_reason TEXT,
                message_count INTEGER DEFAULT 0,
                tool_call_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0,
                cwd TEXT,
                git_branch TEXT,
                git_repo_root TEXT,
                billing_provider TEXT,
                billing_base_url TEXT,
                billing_mode TEXT,
                estimated_cost_usd REAL,
                actual_cost_usd REAL,
                cost_status TEXT,
                cost_source TEXT,
                pricing_version TEXT,
                title TEXT,
                api_call_count INTEGER DEFAULT 0,
                handoff_state TEXT,
                handoff_platform TEXT,
                handoff_error TEXT,
                rewind_count INTEGER NOT NULL DEFAULT 0,
                archived INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_name TEXT,
                timestamp REAL NOT NULL DEFAULT 0,
                token_count INTEGER,
                finish_reason TEXT,
                reasoning TEXT,
                reasoning_content TEXT,
                reasoning_details TEXT,
                codex_reasoning_items TEXT,
                codex_message_items TEXT,
                platform_message_id TEXT,
                observed INTEGER DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                compacted INTEGER NOT NULL DEFAULT 0
            );",
        )
        .context("failed to create tables")?;

        // Insert session metadata row.
        let cwd_str = session
            .workspace
            .as_ref()
            .and_then(|w| w.to_str())
            .unwrap_or("/tmp");

        let started_at_f = session
            .started_at
            .map(|ms| ms as f64 / 1000.0)
            .unwrap_or_else(|| chrono::Utc::now().timestamp() as f64);
        let ended_at_f = session.ended_at.map(|ms| ms as f64 / 1000.0);

        let title_str = session
            .title
            .clone()
            .or_else(|| {
                session
                    .messages
                    .iter()
                    .find(|m| m.role == MessageRole::User)
                    .map(|m| truncate_title(&m.content, 100))
            })
            .unwrap_or_else(|| "Converted Session".to_string());

        conn.execute(
            "INSERT OR REPLACE INTO sessions
                (id, source, model, cwd, title, started_at, ended_at, message_count)
             VALUES (?1, 'casr', ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                target_session_id,
                session.model_name.as_deref().unwrap_or("unknown"),
                cwd_str,
                title_str,
                started_at_f,
                ended_at_f,
                session.messages.len() as i64,
            ],
        )
        .context("failed to insert session row")?;

        // Insert each canonical message as a DB row.
        for msg in &session.messages {
            let role_str = match msg.role {
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
                MessageRole::Tool => "tool",
                MessageRole::System => "system",
                MessageRole::Other(ref r) => r.as_str(),
            };

            let ts_f = msg
                .timestamp
                .map(|ms| ms as f64 / 1000.0)
                .unwrap_or(started_at_f);

            // Serialize tool_calls to JSON for the tool_calls column.
            let tool_calls_json = if msg.tool_calls.is_empty() {
                None
            } else {
                Some(serde_json::to_string(&msg.tool_calls).unwrap_or_default())
            };

            // For tool messages, use the first tool result's call_id.
            let (tool_call_id_opt, tool_name_opt) =
                if msg.role == MessageRole::Tool && !msg.tool_results.is_empty() {
                    let tr = &msg.tool_results[0];
                    (tr.call_id.clone(), None::<String>)
                } else {
                    (None, None)
                };

            conn.execute(
                "INSERT INTO messages
                    (session_id, role, content, tool_call_id, tool_calls, tool_name, timestamp, active)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 1)",
                rusqlite::params![
                    target_session_id,
                    role_str,
                    msg.content,
                    tool_call_id_opt,
                    tool_calls_json,
                    tool_name_opt,
                    ts_f,
                ],
            )
            .context("failed to insert message row")?;
        }

        info!(
            session_id = target_session_id,
            messages = session.messages.len(),
            "Hermes session written to DB"
        );

        Ok(())
    }

    /// Query the DB for all session IDs, returned in started_at-desc order.
    fn list_session_ids(db_path: &Path) -> anyhow::Result<Vec<String>> {
        let conn = Self::open_db(db_path)?;
        let mut stmt = conn
            .prepare("SELECT id FROM sessions ORDER BY started_at DESC")
            .context("failed to prepare list query")?;
        let ids: Vec<String> = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .filter_map(Result::ok)
            .collect();
        Ok(ids)
    }
}

impl Provider for Hermes {
    fn name(&self) -> &str {
        "Hermes"
    }

    fn slug(&self) -> &str {
        "hermes"
    }

    fn cli_alias(&self) -> &str {
        "her"
    }

    fn detect(&self) -> DetectionResult {
        let mut evidence = Vec::new();
        let mut installed = false;

        if let Ok(home) = std::env::var("HERMES_HOME") {
            evidence.push(format!("HERMES_HOME={home}"));
            let db = PathBuf::from(&home).join("state.db");
            if db.is_file() {
                installed = true;
                evidence.push(format!("{} exists", db.display()));
            } else {
                evidence.push(format!("{} missing", db.display()));
            }
        }

        if let Some(db) = Self::db_path().filter(|db| db.is_file()) {
            installed = true;
            evidence.push(format!("{} detected", db.display()));
        }

        // Check for the hermes CLI binary as secondary evidence.
        if which::which("hermes").is_ok() {
            evidence.push("hermes binary found in PATH".to_string());
            // Even if we didn't find the DB yet, if the binary is present
            // the provider may still be usable after first run.
        }

        trace!(provider = "hermes", installed, ?evidence, "detection");
        DetectionResult {
            installed,
            version: None,
            evidence,
        }
    }

    fn session_roots(&self) -> Vec<PathBuf> {
        match Self::db_path() {
            Some(db) if db.is_file() => vec![db],
            _ => vec![],
        }
    }

    fn owns_session(&self, session_id: &str) -> Option<PathBuf> {
        let db_path = Self::db_path()?;
        if !db_path.is_file() {
            return None;
        }
        let conn = Self::open_db(&db_path).ok()?;
        let exists: bool = conn
            .query_row(
                "SELECT 1 FROM sessions WHERE id = ?1",
                rusqlite::params![session_id],
                |_| Ok(true),
            )
            .unwrap_or(false);
        if exists {
            debug!(session_id, "Hermes session found");
            Some(Self::build_virtual_path(&db_path, session_id))
        } else {
            None
        }
    }

    fn read_session(&self, path: &Path) -> anyhow::Result<CanonicalSession> {
        let (db_path, opt_session_id) = Self::parse_virtual_path(path);

        if let Some(session_id) = opt_session_id {
            return Self::read_session_from_db(&db_path, &session_id);
        }

        // Bare DB path without a virtual session ID — try to infer which
        // session was meant from the filename stem (for direct `--source` usage).
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if !stem.is_empty() && path.is_file() {
            let conn = Self::open_db(path).ok();
            if let Some(conn) = conn {
                let exists: bool = conn
                    .query_row(
                        "SELECT 1 FROM sessions WHERE id = ?1",
                        rusqlite::params![stem],
                        |_| Ok(true),
                    )
                    .unwrap_or(false);
                if exists {
                    return Self::read_session_from_db(path, stem);
                }
            }
            // No matching session — list available IDs for a helpful error.
            if let Ok(ids) = Self::list_session_ids(path)
                && !ids.is_empty()
            {
                anyhow::bail!(
                    "Hermes DB at {} contains {} session(s). To read one, use \
                         virtual path format `<db>#<session-id>`. Available IDs: {}",
                    path.display(),
                    ids.len(),
                    ids.join(", "),
                );
            }
        }

        anyhow::bail!(
            "Hermes session path must be in virtual format `<db>#<session_id>` or \
             point to an existing Hermes state.db: {}",
            path.display()
        );
    }

    fn write_session(
        &self,
        session: &CanonicalSession,
        opts: &WriteOptions,
    ) -> anyhow::Result<WrittenSession> {
        let db_path = Self::db_path().ok_or_else(|| {
            anyhow::anyhow!(
                "cannot determine Hermes home directory. Set HERMES_HOME env var \
                 or ensure ~/.hermes exists."
            )
        })?;

        // Use caller-supplied ID when available, otherwise derive one.
        let target_session_id = opts.target_session_id.clone().unwrap_or_else(|| {
            // Deterministic ID based on timestamp.
            let now = chrono::Utc::now();
            format!(
                "{}-{}{}",
                now.format("%Y%m%dT%H%M%S"),
                &uuid::Uuid::new_v4().to_string()[..8],
                CASR_ID_SUFFIX,
            )
        });

        Self::write_session_to_db(&db_path, session, &target_session_id)?;

        let virtual_path = Self::build_virtual_path(&db_path, &target_session_id);

        debug!(session_id = target_session_id, "Hermes session written");

        Ok(WrittenSession {
            paths: vec![virtual_path],
            session_id: target_session_id.clone(),
            resume_command: self.resume_command(&target_session_id),
            backup_path: None,
            warnings: vec![],
        })
    }

    fn resume_command(&self, session_id: &str) -> String {
        format!("hermes --resume {session_id}")
    }

    fn list_sessions(&self) -> Option<Vec<(String, PathBuf)>> {
        let db_path = Self::db_path()?;
        if !db_path.is_file() {
            return Some(vec![]);
        }
        let ids = Self::list_session_ids(&db_path).ok()?;
        let sessions: Vec<(String, PathBuf)> = ids
            .into_iter()
            .map(|id| {
                let vpath = Self::build_virtual_path(&db_path, &id);
                (id, vpath)
            })
            .collect();

        debug!(count = sessions.len(), "Hermes list_sessions");
        Some(sessions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::Provider;
    use serde_json::json;
    use std::path::PathBuf;

    /// Create a test Hermes DB with the given session and return the Db path.
    #[allow(clippy::type_complexity)]
    fn create_test_db(
        session_id: &str,
        source: &str,
        model: Option<&str>,
        cwd: Option<&str>,
        title: Option<&str>,
        started_at: Option<f64>,
        messages: &[(i64, &str, &str, Option<&str>, Option<&str>)],
    ) -> (tempfile::TempDir, PathBuf) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("state.db");

        let conn = rusqlite::Connection::open(&db_path).expect("open db");
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, source TEXT, model TEXT, cwd TEXT,
                title TEXT, started_at REAL, ended_at REAL,
                message_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL, role TEXT NOT NULL,
                content TEXT, tool_call_id TEXT, tool_calls TEXT,
                tool_name TEXT, timestamp REAL,
                reasoning TEXT, reasoning_content TEXT,
                active INTEGER NOT NULL DEFAULT 1
            );",
        )
        .expect("create tables");

        conn.execute(
            "INSERT INTO sessions (id, source, model, cwd, title, started_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![session_id, source, model, cwd, title, started_at,],
        )
        .expect("insert session");

        for (msg_id, role, content, tool_calls, tool_call_id) in messages {
            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, tool_call_id, tool_calls, timestamp, active)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 1)",
                rusqlite::params![
                    msg_id,
                    session_id,
                    role,
                    content,
                    tool_call_id,
                    tool_calls,
                    started_at.unwrap_or(1_700_000_000.0) + *msg_id as f64,
                ],
            )
            .expect("insert message");
        }

        (tmp, db_path)
    }

    fn make_canonical_session(messages: Vec<CanonicalMessage>) -> CanonicalSession {
        CanonicalSession {
            session_id: "test-session-1".to_string(),
            provider_slug: "hermes".to_string(),
            workspace: Some(PathBuf::from("/data/projects/test_ws")),
            title: Some("Test Session".to_string()),
            started_at: Some(1_700_000_000_000),
            ended_at: Some(1_700_000_001_000),
            messages,
            metadata: serde_json::Value::Null,
            source_path: PathBuf::from("/tmp/source.jsonl"),
            model_name: None,
        }
    }

    // -----------------------------------------------------------------------
    // Provider metadata tests
    // -----------------------------------------------------------------------

    #[test]
    fn provider_metadata() {
        let provider = Hermes;
        assert_eq!(provider.name(), "Hermes");
        assert_eq!(provider.slug(), "hermes");
        assert_eq!(provider.cli_alias(), "her");
    }

    #[test]
    fn resume_command() {
        assert_eq!(Hermes.resume_command("abc-123"), "hermes --resume abc-123");
    }

    // -----------------------------------------------------------------------
    // Virtual path tests
    // -----------------------------------------------------------------------

    #[test]
    fn virtual_path_roundtrip() {
        let db = PathBuf::from("/home/user/.hermes/state.db");
        let sid = "session-001";
        let vpath = Hermes::build_virtual_path(&db, sid);
        let (parsed_db, parsed_sid) = Hermes::parse_virtual_path(&vpath);
        assert_eq!(parsed_db, db);
        assert_eq!(parsed_sid, Some(sid.to_string()));
    }

    #[test]
    fn virtual_path_no_separator() {
        let db = PathBuf::from("/home/user/.hermes/state.db");
        let (parsed_db, parsed_sid) = Hermes::parse_virtual_path(&db);
        assert_eq!(parsed_db, db);
        assert!(parsed_sid.is_none());
    }

    #[test]
    fn virtual_path_hash_in_db_name() {
        // Ensure a hash in the filename doesn't confuse parsing.
        let vpath = PathBuf::from("/home/user/.hermes/state#1.db#session-001");
        let (parsed_db, parsed_sid) = Hermes::parse_virtual_path(&vpath);
        assert_eq!(
            parsed_db.display().to_string(),
            "/home/user/.hermes/state#1.db"
        );
        assert_eq!(parsed_sid, Some("session-001".to_string()));
    }

    /// Set the Hermes home directory for tests (thread-local override).
    /// Uses `TEST_HERMES_HOME` thread-local instead of env vars to avoid
    /// `unsafe` on Rust 2024.
    fn set_test_home(dir: &Path) -> impl Drop {
        let prev = TEST_HERMES_HOME.with(|cell| cell.replace(Some(dir.to_path_buf())));
        struct Restore(Option<PathBuf>);
        impl Drop for Restore {
            fn drop(&mut self) {
                TEST_HERMES_HOME.with(|cell| {
                    cell.replace(self.0.take());
                });
            }
        }
        Restore(prev)
    }

    // -----------------------------------------------------------------------
    // owns_session tests
    // -----------------------------------------------------------------------

    #[test]
    fn owns_session_found() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let _guard = set_test_home(tmp.path());
        let db_path = tmp.path().join("state.db");
        let conn = rusqlite::Connection::open(&db_path).expect("open db");
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, source TEXT);",
        )
        .expect("create table");
        conn.execute(
            "INSERT INTO sessions (id, source) VALUES (?1, 'cli')",
            rusqlite::params!["session-abc"],
        )
        .expect("insert");

        let result = Hermes.owns_session("session-abc");
        assert!(result.is_some(), "session should be found");
    }

    #[test]
    fn owns_session_not_found() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let _guard = set_test_home(tmp.path());
        let db_path = tmp.path().join("state.db");
        let conn = rusqlite::Connection::open(&db_path).expect("open db");
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, source TEXT);",
        )
        .expect("create table");

        let result = Hermes.owns_session("nonexistent");
        assert!(result.is_none(), "nonexistent session should not be found");
    }

    #[test]
    fn owns_session_no_db() {
        // No DB at all — should return None.
        let result = Hermes.owns_session("anything");
        assert!(result.is_none(), "no DB should return None");
    }

    // -----------------------------------------------------------------------
    // list_sessions tests
    // -----------------------------------------------------------------------

    #[test]
    fn list_sessions_empty_when_no_db() {
        // Use a temp dir as HERMES_HOME that has no state.db.
        let tmp = tempfile::tempdir().expect("tempdir");
        let _guard = set_test_home(tmp.path());
        let result = Hermes.list_sessions();
        assert_eq!(result, Some(vec![]));
    }

    #[test]
    fn list_sessions_returns_all_sessions() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let _guard = set_test_home(tmp.path());
        let db_path = tmp.path().join("state.db");
        let conn = rusqlite::Connection::open(&db_path).expect("open db");
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, source TEXT, started_at REAL);",
        )
        .expect("create table");
        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES ('s1', 'cli', 1000.0)",
            [],
        )
        .expect("insert s1");
        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES ('s2', 'cli', 2000.0)",
            [],
        )
        .expect("insert s2");

        let sessions = Hermes.list_sessions().expect("list_sessions");
        assert_eq!(sessions.len(), 2);
        // Should be ordered by started_at DESC.
        assert_eq!(sessions[0].0, "s2");
        assert_eq!(sessions[1].0, "s1");
    }

    // -----------------------------------------------------------------------
    // Reader tests
    // -----------------------------------------------------------------------

    #[test]
    fn reader_basic_exchange() {
        let messages = vec![
            (1, "user", "Hello Hermes!", None, None),
            (2, "assistant", "Hi there!", None, None),
        ];
        let (_tmp, db_path) = create_test_db(
            "session-001",
            "cli",
            Some("claude-3.5-sonnet"),
            Some("/data/project"),
            None,
            Some(1_700_000_000.0),
            &messages,
        );

        let vpath = Hermes::build_virtual_path(&db_path, "session-001");
        let session = Hermes.read_session(&vpath).expect("read_session");

        assert_eq!(session.provider_slug, "hermes");
        assert_eq!(session.session_id, "session-001");
        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[0].role, MessageRole::User);
        assert_eq!(session.messages[0].content, "Hello Hermes!");
        assert_eq!(session.messages[1].role, MessageRole::Assistant);
        assert_eq!(session.messages[1].content, "Hi there!");
        assert_eq!(session.workspace, Some(PathBuf::from("/data/project")));
        assert_eq!(session.model_name.as_deref(), Some("claude-3.5-sonnet"));
        assert!(session.started_at.is_some());
    }

    #[test]
    fn reader_with_title() {
        let messages = vec![
            (1, "user", "Fix the bug", None, None),
            (2, "assistant", "Working on it", None, None),
        ];
        let (_tmp, db_path) = create_test_db(
            "session-002",
            "cli",
            None,
            None,
            Some("My Hermes Session"),
            Some(1_700_000_000.0),
            &messages,
        );

        let vpath = Hermes::build_virtual_path(&db_path, "session-002");
        let session = Hermes.read_session(&vpath).expect("read_session");

        assert_eq!(session.title.as_deref(), Some("My Hermes Session"));
    }

    #[test]
    fn reader_title_fallback_to_first_user_message() {
        let messages = vec![
            (1, "user", "Implement dark mode", None, None),
            (2, "assistant", "OK!", None, None),
        ];
        let (_tmp, db_path) = create_test_db(
            "session-003",
            "cli",
            None,
            None,
            None,
            Some(1_700_000_000.0),
            &messages,
        );

        let vpath = Hermes::build_virtual_path(&db_path, "session-003");
        let session = Hermes.read_session(&vpath).expect("read_session");

        assert_eq!(session.title.as_deref(), Some("Implement dark mode"));
    }

    #[test]
    fn reader_tool_calls_parsed() {
        let tool_calls_json = json!([
            {"id": "call-1", "name": "Bash", "arguments": {"command": "ls"}}
        ])
        .to_string();

        let messages = vec![
            (1, "user", "Run ls", None, None),
            (
                2,
                "assistant",
                "Running it",
                Some(tool_calls_json.as_str()),
                None,
            ),
        ];
        let (_tmp, db_path) = create_test_db(
            "session-004",
            "cli",
            None,
            None,
            None,
            Some(1_700_000_000.0),
            &messages,
        );

        let vpath = Hermes::build_virtual_path(&db_path, "session-004");
        let session = Hermes.read_session(&vpath).expect("read_session");

        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[1].tool_calls.len(), 1);
        assert_eq!(session.messages[1].tool_calls[0].name, "Bash");
        assert_eq!(session.messages[1].tool_calls[0].arguments["command"], "ls");
    }

    #[test]
    fn reader_tool_results() {
        let messages = vec![
            (1, "user", "Run command", None, None),
            (2, "assistant", "Running", None, None),
            (3, "tool", "{\"stdout\": \"done\"}", None, Some("call-1")),
        ];
        let (_tmp, db_path) = create_test_db(
            "session-005",
            "cli",
            None,
            None,
            None,
            Some(1_700_000_000.0),
            &messages,
        );

        let vpath = Hermes::build_virtual_path(&db_path, "session-005");
        let session = Hermes.read_session(&vpath).expect("read_session");

        let tool_msg = session
            .messages
            .iter()
            .find(|m| m.role == MessageRole::Tool)
            .expect("should have a tool message");
        assert_eq!(tool_msg.tool_results.len(), 1);
        assert_eq!(tool_msg.tool_results[0].call_id.as_deref(), Some("call-1"));
        assert!(tool_msg.content.contains("done"));
    }

    #[test]
    fn reader_skips_inactive_messages() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("state.db");
        {
            let conn = rusqlite::Connection::open(&db_path).expect("open db");
            conn.execute_batch(
                "CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, model TEXT, cwd TEXT, title TEXT,
                    started_at REAL, ended_at REAL, message_count INTEGER DEFAULT 0,
                    input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0,
                    reasoning_tokens INTEGER DEFAULT 0);
                 CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT, role TEXT, content TEXT, timestamp REAL,
                    tool_call_id TEXT, tool_calls TEXT, tool_name TEXT,
                    reasoning TEXT, reasoning_content TEXT,
                    active INTEGER NOT NULL DEFAULT 1);",
            )
            .expect("create tables");

            conn.execute(
                "INSERT INTO sessions (id, source, started_at) VALUES ('s1', 'cli', 1000.0)",
                [],
            )
            .expect("insert session");

            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, active)
                 VALUES ('s1', 'user', 'Active msg', 1001.0, 1)",
                [],
            )
            .expect("insert active msg");
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, active)
                 VALUES ('s1', 'assistant', 'Skipped msg', 1002.0, 0)",
                [],
            )
            .expect("insert inactive msg");
        }

        let vpath = Hermes::build_virtual_path(&db_path, "s1");
        let session = Hermes.read_session(&vpath).expect("read_session");

        assert_eq!(session.messages.len(), 1);
        assert_eq!(session.messages[0].content, "Active msg");
    }

    #[test]
    fn reader_empty_session_no_messages() {
        let (_tmp, db_path) = create_test_db(
            "session-empty",
            "cli",
            None,
            None,
            None,
            Some(1_700_000_000.0),
            &[],
        );

        let vpath = Hermes::build_virtual_path(&db_path, "session-empty");
        let session = Hermes.read_session(&vpath).expect("read_session");

        assert!(session.messages.is_empty());
        assert_eq!(session.session_id, "session-empty");
    }

    #[test]
    fn reader_session_not_found_error() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("state.db");
        let conn = rusqlite::Connection::open(&db_path).expect("open db");
        conn.execute_batch("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT);")
            .expect("create table");

        let vpath = Hermes::build_virtual_path(&db_path, "nonexistent");
        let err = Hermes.read_session(&vpath).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("not found"),
            "expected 'not found' error, got: {msg}"
        );
    }

    #[test]
    fn reader_bare_db_path_with_no_session() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("state.db");
        let conn = rusqlite::Connection::open(&db_path).expect("open db");
        conn.execute_batch("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT);")
            .expect("create table");

        let err = Hermes.read_session(&db_path).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("virtual format"),
            "expected 'virtual format' error, got: {msg}"
        );
    }

    #[test]
    fn reader_bare_db_path_with_existing_session_id_match() {
        let messages = vec![(1, "user", "Hello", None, None)];
        let (_tmp, db_path) = create_test_db(
            "state", // session ID == filename stem
            "cli",
            None,
            None,
            None,
            Some(1_700_000_000.0),
            &messages,
        );

        // Reading the bare DB path with stem matching a session ID should work.
        let session = Hermes.read_session(&db_path).expect("read_session");
        assert_eq!(session.session_id, "state");
        assert_eq!(session.messages.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Writer tests
    // -----------------------------------------------------------------------

    #[test]
    fn writer_creates_db_and_inserts_session() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let _guard = set_test_home(tmp.path());
        let session = make_canonical_session(vec![
            CanonicalMessage {
                idx: 0,
                role: MessageRole::User,
                content: "Hello from writer".to_string(),
                timestamp: Some(1_700_000_000_000),
                author: None,
                tool_calls: vec![],
                tool_results: vec![],
                extra: serde_json::Value::Null,
            },
            CanonicalMessage {
                idx: 1,
                role: MessageRole::Assistant,
                content: "Hello back".to_string(),
                timestamp: Some(1_700_000_001_000),
                author: None,
                tool_calls: vec![],
                tool_results: vec![],
                extra: serde_json::Value::Null,
            },
        ]);

        let written = Hermes
            .write_session(
                &session,
                &WriteOptions {
                    force: false,
                    target_session_id: Some("test-written-session".to_string()),
                },
            )
            .expect("write_session");

        assert_eq!(written.session_id, "test-written-session");
        assert_eq!(written.paths.len(), 1);
        assert_eq!(
            written.resume_command,
            "hermes --resume test-written-session"
        );

        // Verify the DB was created and the session is readable.
        let readback = Hermes.read_session(&written.paths[0]).expect("readback");
        assert_eq!(readback.messages.len(), 2);
        assert_eq!(readback.messages[0].content, "Hello from writer");
        assert_eq!(readback.messages[1].content, "Hello back");
        assert_eq!(readback.session_id, "test-written-session");
    }

    #[test]
    fn writer_create_tables_on_live_db() {
        // Hermes's native DB has the full schema. When we write to an
        // existing DB, the CREATE TABLE IF NOT EXISTS should be a no-op
        // and the INSERT should succeed.
        let tmp = tempfile::tempdir().expect("tempdir");
        let _guard = set_test_home(tmp.path());
        let db_path = tmp.path().join("state.db");

        // Pre-create a DB with the full Hermes schema.
        let conn = rusqlite::Connection::open(&db_path).expect("open db");
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, source TEXT NOT NULL, model TEXT,
                cwd TEXT, title TEXT, started_at REAL, ended_at REAL,
                message_count INTEGER DEFAULT 0, tool_call_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0, user_id TEXT,
                model_config TEXT, system_prompt TEXT, parent_session_id TEXT,
                end_reason TEXT, cache_read_tokens INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0, git_branch TEXT,
                git_repo_root TEXT, billing_provider TEXT, billing_base_url TEXT,
                billing_mode TEXT, estimated_cost_usd REAL, actual_cost_usd REAL,
                cost_status TEXT, cost_source TEXT, pricing_version TEXT,
                api_call_count INTEGER DEFAULT 0, handoff_state TEXT,
                handoff_platform TEXT, handoff_error TEXT,
                rewind_count INTEGER NOT NULL DEFAULT 0,
                archived INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL, role TEXT NOT NULL,
                    content TEXT, tool_call_id TEXT, tool_calls TEXT,
                    tool_name TEXT, timestamp REAL NOT NULL,
                    token_count INTEGER, finish_reason TEXT,
                    reasoning TEXT, reasoning_content TEXT, reasoning_details TEXT,
                    codex_reasoning_items TEXT, codex_message_items TEXT,
                    platform_message_id TEXT, observed INTEGER DEFAULT 0,
                    active INTEGER NOT NULL DEFAULT 1, compacted INTEGER NOT NULL DEFAULT 0
                );",
        )
        .expect("create full schema");

        let session = make_canonical_session(vec![CanonicalMessage {
            idx: 0,
            role: MessageRole::User,
            content: "Test on existing DB".to_string(),
            timestamp: Some(1_700_000_000_000),
            author: None,
            tool_calls: vec![],
            tool_results: vec![],
            extra: serde_json::Value::Null,
        }]);

        let written = Hermes
            .write_session(
                &session,
                &WriteOptions {
                    force: false,
                    target_session_id: Some("existing-db-test".to_string()),
                },
            )
            .expect("write_session on existing DB");

        assert_eq!(written.session_id, "existing-db-test");

        // Read back.
        let readback = Hermes
            .read_session(&written.paths[0])
            .expect("readback from existing DB");
        assert_eq!(readback.messages[0].content, "Test on existing DB");
    }

    #[test]
    fn writer_tool_calls_preserved() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let _guard = set_test_home(tmp.path());
        let session = make_canonical_session(vec![CanonicalMessage {
            idx: 0,
            role: MessageRole::Assistant,
            content: "Let me check".to_string(),
            timestamp: Some(1_700_000_000_000),
            author: None,
            tool_calls: vec![ToolCall {
                id: Some("tc-1".to_string()),
                name: "Bash".to_string(),
                arguments: json!({"command": "ls -la"}),
            }],
            tool_results: vec![],
            extra: serde_json::Value::Null,
        }]);

        let written = Hermes
            .write_session(
                &session,
                &WriteOptions {
                    force: false,
                    target_session_id: Some("tool-call-test".to_string()),
                },
            )
            .expect("write_session");

        let readback = Hermes.read_session(&written.paths[0]).expect("readback");
        assert_eq!(readback.messages[0].tool_calls.len(), 1);
        assert_eq!(readback.messages[0].tool_calls[0].name, "Bash");
        assert_eq!(
            readback.messages[0].tool_calls[0].arguments["command"],
            "ls -la"
        );
    }
}

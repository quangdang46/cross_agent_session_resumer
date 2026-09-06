//! ZCode provider — reads/writes sessions from SQLite `db.sqlite`.
//!
//! ZCode (Z.AI's ADE) stores sessions in a single SQLite database at
//! `~/.zcode/cli/db/db.sqlite` (override with `$ZCODE_HOME/cli/db/db.sqlite`
//! or `$ZCODE_DB_PATH`). The schema is the V2 layout shared with OpenCode:
//! tables `session` / `message` / `part` with JSON `data` columns.
//!
//! Session IDs are plain UUIDs (no prefix). casr addresses sessions via a
//! virtual path form: `<db-path>/<urlencoded-session-id>`.
//!
//! Sub-agent child sessions (`task_type = 'subagent_child'`) are excluded from
//! listing by default to keep the session list focused on interactive work.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::Context;
use rusqlite::{Connection, OpenFlags};
use tracing::{debug, info, trace};

#[cfg(test)]
thread_local! {
    /// Unit-test override for the ZCode DB path (avoids process-wide env mutation).
    static TEST_DB_PATH_OVERRIDE: std::cell::RefCell<Option<PathBuf>> =
        const { std::cell::RefCell::new(None) };
}

use crate::discovery::DetectionResult;
use crate::model::{
    CanonicalMessage, CanonicalSession, MessageRole, ToolCall, ToolResult, flatten_content,
    normalize_role, parse_timestamp, reindex_messages, truncate_title,
};
use crate::providers::{Provider, WriteOptions, WrittenSession};

/// ZCode provider implementation.
pub struct ZCode;

const DB_FILENAME: &str = "db.sqlite";
const DB_DIRNAME: &str = "db";
const CLI_DIRNAME: &str = "cli";
const ZCODE_DIRNAME: &str = ".zcode";

impl ZCode {
    /// Parse ZCODE environment overrides into a target DB path.
    ///
    /// Supported overrides:
    /// - unit-test thread-local (`TEST_DB_PATH_OVERRIDE`)
    /// - `ZCODE_DB_PATH` (direct file path)
    /// - `ZCODE_HOME` (directory containing `cli/db/db.sqlite`, or a direct
    ///   `.sqlite`/`.db` path)
    fn env_db_path() -> Option<PathBuf> {
        #[cfg(test)]
        {
            if let Some(path) = TEST_DB_PATH_OVERRIDE.with(|cell| cell.borrow().clone()) {
                return Some(path);
            }
        }

        if let Ok(path) = std::env::var("ZCODE_DB_PATH")
            && !path.trim().is_empty()
        {
            return Some(PathBuf::from(path));
        }

        if let Ok(home) = std::env::var("ZCODE_HOME")
            && !home.trim().is_empty()
        {
            let home_path = PathBuf::from(home);
            // Direct DB file path.
            if home_path
                .extension()
                .is_some_and(|ext| ext == "db" || ext == "sqlite")
            {
                return Some(home_path);
            }
            // Directory: assume `<home>/cli/db/db.sqlite`.
            return Some(home_path.join(CLI_DIRNAME).join(DB_DIRNAME).join(DB_FILENAME));
        }

        None
    }

    /// Candidate DB paths for the ZCode global database.
    ///
    /// ZCode stores its DB at `~/.zcode/cli/db/db.sqlite`.
    fn global_db_candidates() -> Vec<PathBuf> {
        let mut paths = Vec::new();
        if let Some(home) = dirs::home_dir() {
            paths.push(
                home.join(ZCODE_DIRNAME)
                    .join(CLI_DIRNAME)
                    .join(DB_DIRNAME)
                    .join(DB_FILENAME),
            );
        }
        dedup_existing_files(paths)
    }

    /// First existing global DB, if any.
    fn global_db_path() -> Option<PathBuf> {
        Self::global_db_candidates()
            .into_iter()
            .find(|p| p.is_file())
    }

    /// Discover existing ZCode DB files.
    ///
    /// If env override is set, discovery is constrained to that location.
    fn find_db_files() -> Vec<PathBuf> {
        if let Some(env_db) = Self::env_db_path() {
            return if env_db.is_file() {
                vec![env_db]
            } else {
                Vec::new()
            };
        }

        dedup_existing_files(Self::global_db_candidates())
    }

    /// Resolve target DB path for writes.
    ///
    /// Priority:
    /// 1. `ZCODE_DB_PATH` / `ZCODE_HOME`
    /// 2. Existing global DB (`~/.zcode/cli/db/db.sqlite`)
    /// 3. First available from candidates
    fn choose_target_db_path() -> anyhow::Result<PathBuf> {
        if let Some(env_db) = Self::env_db_path() {
            return Ok(env_db);
        }

        if let Some(global) = Self::global_db_path() {
            return Ok(global);
        }

        if let Some(existing) = Self::find_db_files().into_iter().next() {
            return Ok(existing);
        }

        // Create default location.
        if let Some(home) = dirs::home_dir() {
            let default = home
                .join(ZCODE_DIRNAME)
                .join(CLI_DIRNAME)
                .join(DB_DIRNAME)
                .join(DB_FILENAME);
            if let Some(parent) = default.parent() {
                std::fs::create_dir_all(parent)
                    .context("failed to create ZCode db directory")?;
            }
            return Ok(default);
        }

        anyhow::bail!("cannot determine ZCode DB path: no home directory and no env override")
    }

    fn table_exists(conn: &Connection, table: &str) -> bool {
        conn.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1")
            .and_then(|mut stmt| stmt.exists(rusqlite::params![table]))
            .unwrap_or(false)
    }

    fn detect_schema(conn: &Connection) -> SchemaKind {
        if Self::table_exists(conn, "session") && Self::table_exists(conn, "message") {
            SchemaKind::V2
        } else {
            SchemaKind::Legacy
        }
    }

    fn ensure_zc_prefix(id: &str) -> String {
        if id.starts_with("zc_") {
            id.to_string()
        } else {
            format!("zc_{id}")
        }
    }

    fn mint_entity_id(prefix: &str) -> String {
        format!("{prefix}{}", uuid::Uuid::new_v4().simple())
    }

    /// Build virtual per-session path: `<db-path>/<urlencoded-session-id>`.
    fn virtual_session_path(db_path: &Path, session_id: &str) -> PathBuf {
        let encoded = urlencoding::encode(session_id);
        db_path.join(encoded.as_ref())
    }

    /// Parse virtual path back into `(db_path, session_id)`.
    fn parse_virtual_path(path: &Path) -> Option<(PathBuf, String)> {
        let parent = path.parent()?;
        if !parent.is_file() {
            return None;
        }
        let parent_name = parent.file_name()?.to_str()?;
        if parent_name != DB_FILENAME {
            return None;
        }

        let encoded = path.file_name()?.to_str()?;
        let decoded = urlencoding::decode(encoded).ok()?;
        Some((parent.to_path_buf(), decoded.into_owned()))
    }

    /// Open DB in read-only mode.
    fn open_db(path: &Path) -> anyhow::Result<Connection> {
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("failed to open ZCode DB: {}", path.display()))?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        Ok(conn)
    }

    /// Open DB in read-write/create mode.
    fn open_db_rw(path: &Path) -> anyhow::Result<Connection> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directory: {}", parent.display()))?;
        }

        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("failed to open ZCode DB for writing: {}", path.display()))?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        Ok(conn)
    }

    fn session_exists(conn: &Connection, session_id: &str) -> bool {
        match Self::detect_schema(conn) {
            SchemaKind::V2 => conn
                .prepare("SELECT 1 FROM session WHERE id = ?1 LIMIT 1")
                .and_then(|mut stmt| stmt.exists(rusqlite::params![session_id]))
                .unwrap_or(false),
            SchemaKind::Legacy => {
                if !Self::table_exists(conn, "sessions") {
                    return false;
                }
                conn.prepare("SELECT 1 FROM sessions WHERE id = ?1 LIMIT 1")
                    .and_then(|mut stmt| stmt.exists(rusqlite::params![session_id]))
                    .unwrap_or(false)
            }
        }
    }

    fn newest_root_session_id(conn: &Connection) -> Option<String> {
        match Self::detect_schema(conn) {
            SchemaKind::V2 => conn
                .query_row(
                    "SELECT id FROM session WHERE parent_id IS NULL
                     ORDER BY time_created DESC LIMIT 1",
                    [],
                    |row| row.get(0),
                )
                .ok(),
            SchemaKind::Legacy => {
                if !Self::table_exists(conn, "sessions") {
                    return None;
                }
                conn.query_row(
                    "SELECT id FROM sessions WHERE parent_session_id IS NULL
                     ORDER BY created_at DESC LIMIT 1",
                    [],
                    |row| row.get(0),
                )
                .ok()
            }
        }
    }

    fn read_session_by_id(
        conn: &Connection,
        db_path: &Path,
        session_id: &str,
    ) -> anyhow::Result<CanonicalSession> {
        match Self::detect_schema(conn) {
            SchemaKind::V2 => Self::read_session_v2(conn, db_path, session_id),
            SchemaKind::Legacy => Self::read_session_legacy(conn, db_path, session_id),
        }
    }

    fn read_session_legacy(
        conn: &Connection,
        db_path: &Path,
        session_id: &str,
    ) -> anyhow::Result<CanonicalSession> {
        if !Self::table_exists(conn, "sessions") {
            anyhow::bail!("ZCode DB has no sessions table: {}", db_path.display());
        }
        if !Self::table_exists(conn, "messages") {
            anyhow::bail!("ZCode DB has no messages table: {}", db_path.display());
        }

        let (title_raw, created_raw, updated_raw, parent_session_id): (String, i64, i64, Option<String>) = conn
            .query_row(
                "SELECT title, created_at, updated_at, parent_session_id
                 FROM sessions
                 WHERE id = ?1
                 LIMIT 1",
                rusqlite::params![session_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .with_context(|| format!("session '{session_id}' not found in {}", db_path.display()))?;

        let mut started_at = parse_timestamp(&serde_json::Value::from(created_raw));
        let mut ended_at = parse_timestamp(&serde_json::Value::from(updated_raw)).or(started_at);
        let mut model_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut messages = Vec::new();

        let mut stmt = conn
            .prepare(
                "SELECT id, role, parts, model, created_at, updated_at
                 FROM messages
                 WHERE session_id = ?1
                 ORDER BY created_at ASC, id ASC",
            )
            .context("failed to prepare message query")?;

        let rows = stmt.query_map(rusqlite::params![session_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, i64>(5)?,
            ))
        })?;

        for row in rows {
            let (message_id, role_raw, parts_json, model, created_at_raw, _updated_at_raw) = row?;

            let timestamp =
                parse_timestamp(&serde_json::Value::from(created_at_raw)).or(Some(created_at_raw));
            if let Some(ts) = timestamp {
                started_at = Some(started_at.map_or(ts, |current| current.min(ts)));
                ended_at = Some(ended_at.map_or(ts, |current| current.max(ts)));
            }

            let raw_parts = serde_json::from_str::<serde_json::Value>(&parts_json)
                .unwrap_or_else(|_| serde_json::json!([]));
            let (content, tool_calls, tool_results) = parse_parts(&raw_parts);

            if let Some(model_name) = model.as_deref().filter(|m| !m.is_empty()) {
                *model_counts.entry(model_name.to_string()).or_insert(0) += 1;
            }

            messages.push(CanonicalMessage {
                idx: 0,
                role: normalize_role(&role_raw),
                content,
                timestamp,
                author: model.clone(),
                tool_calls,
                tool_results,
                extra: serde_json::json!({
                    "zcode_message_id": message_id,
                    "zcode_parts": raw_parts,
                }),
            });
        }

        reindex_messages(&mut messages);

        let title = (!title_raw.trim().is_empty())
            .then_some(title_raw)
            .or_else(|| {
                messages
                    .iter()
                    .find(|m| m.role == MessageRole::User)
                    .map(|m| truncate_title(&m.content, 80))
                    .filter(|t| !t.is_empty())
            });

        let model_name = model_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(name, _)| name);

        let source = Self::virtual_session_path(db_path, session_id);

        Ok(CanonicalSession {
            session_id: session_id.to_string(),
            provider_slug: "zcode".to_string(),
            workspace: None,
            title,
            started_at,
            ended_at,
            messages,
            metadata: serde_json::json!({
                "zcode_db": db_path.display().to_string(),
                "zcode_schema": "legacy",
                "parent_session_id": parent_session_id,
            }),
            source_path: source,
            model_name,
        })
    }

    fn read_session_v2(
        conn: &Connection,
        db_path: &Path,
        session_id: &str,
    ) -> anyhow::Result<CanonicalSession> {
        struct SessionRow {
            title: String,
            directory: String,
            parent_id: Option<String>,
            created_raw: i64,
            updated_raw: i64,
        }
        let row = conn
            .query_row(
                "SELECT title, directory, parent_id, time_created, time_updated
                 FROM session
                 WHERE id = ?1
                 LIMIT 1",
                rusqlite::params![session_id],
                |row| {
                    Ok(SessionRow {
                        title: row.get(0)?,
                        directory: row.get(1)?,
                        parent_id: row.get(2)?,
                        created_raw: row.get(3)?,
                        updated_raw: row.get(4)?,
                    })
                },
            )
            .with_context(|| {
                format!("session '{session_id}' not found in {}", db_path.display())
            })?;
        let SessionRow {
            title: title_raw,
            directory,
            parent_id,
            created_raw,
            updated_raw,
        } = row;

        let mut started_at = parse_timestamp(&serde_json::Value::from(created_raw));
        let mut ended_at = parse_timestamp(&serde_json::Value::from(updated_raw)).or(started_at);
        let mut model_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut messages = Vec::new();

        let mut msg_stmt = conn
            .prepare(
                "SELECT id, data, time_created, time_updated
                 FROM message
                 WHERE session_id = ?1
                 ORDER BY COALESCE(sequence, 0) ASC, time_created ASC, id ASC",
            )
            .context("failed to prepare v2 message query")?;

        let msg_rows = msg_stmt.query_map(rusqlite::params![session_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })?;

        for row in msg_rows {
            let (message_id, data_json, created_at_raw, _updated_at_raw) = row?;
            let data: serde_json::Value =
                serde_json::from_str(&data_json).unwrap_or_else(|_| serde_json::json!({}));
            let role_raw = data
                .get("role")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("other");

            let timestamp =
                parse_timestamp(&serde_json::Value::from(created_at_raw)).or(Some(created_at_raw));
            if let Some(ts) = timestamp {
                started_at = Some(started_at.map_or(ts, |current| current.min(ts)));
                ended_at = Some(ended_at.map_or(ts, |current| current.max(ts)));
            }

            // Load parts for this message.
            let mut part_stmt = conn
                .prepare(
                    "SELECT data FROM part
                     WHERE message_id = ?1
                     ORDER BY COALESCE(sequence, 0) ASC, time_created ASC, id ASC",
                )
                .context("failed to prepare v2 part query")?;
            let part_rows = part_stmt
                .query_map(rusqlite::params![message_id], |row| row.get::<_, String>(0))?;
            let mut parts_arr = Vec::new();
            for prow in part_rows {
                let pjson = prow?;
                if let Ok(pval) = serde_json::from_str::<serde_json::Value>(&pjson) {
                    parts_arr.push(pval);
                }
            }
            let raw_parts = serde_json::Value::Array(parts_arr);
            let (content, tool_calls, tool_results) = parse_parts(&raw_parts);

            // Extract model from message data (ZCode stores modelID).
            let model = data
                .get("modelID")
                .and_then(serde_json::Value::as_str)
                .or_else(|| data.pointer("/model/modelID").and_then(serde_json::Value::as_str))
                .or_else(|| data.pointer("/model/id").and_then(serde_json::Value::as_str))
                .filter(|m| !m.is_empty())
                .map(ToString::to_string);

            if let Some(model_name) = model.as_deref() {
                *model_counts.entry(model_name.to_string()).or_insert(0) += 1;
            }

            messages.push(CanonicalMessage {
                idx: 0,
                role: normalize_role(role_raw),
                content,
                timestamp,
                author: model,
                tool_calls,
                tool_results,
                extra: serde_json::json!({
                    "zcode_message_id": message_id,
                    "zcode_message_data": data,
                    "zcode_parts": raw_parts,
                }),
            });
        }

        reindex_messages(&mut messages);

        let title = (!title_raw.trim().is_empty())
            .then_some(title_raw)
            .or_else(|| {
                messages
                    .iter()
                    .find(|m| m.role == MessageRole::User)
                    .map(|m| truncate_title(&m.content, 80))
                    .filter(|t| !t.is_empty())
            });

        // Prefer per-message models; ZCode session table has no model column,
        // so we rely entirely on per-message data.
        let model_name = model_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(name, _)| name);

        let workspace = if directory.trim().is_empty() {
            None
        } else {
            Some(PathBuf::from(&directory))
        };

        let source = Self::virtual_session_path(db_path, session_id);

        Ok(CanonicalSession {
            session_id: session_id.to_string(),
            provider_slug: "zcode".to_string(),
            workspace,
            title,
            started_at,
            ended_at,
            messages,
            metadata: serde_json::json!({
                "zcode_db": db_path.display().to_string(),
                "zcode_schema": "v2",
                "parent_session_id": parent_id,
                "directory": directory,
            }),
            source_path: source,
            model_name,
        })
    }
}

// ---------------------------------------------------------------------------
// Provider trait implementation
// ---------------------------------------------------------------------------

impl Provider for ZCode {
    fn name(&self) -> &str {
        "ZCode"
    }

    fn slug(&self) -> &str {
        "zcode"
    }

    fn cli_alias(&self) -> &str {
        "zc"
    }

    fn detect(&self) -> DetectionResult {
        let mut installed = false;
        let mut evidence = Vec::new();

        if which::which("zcode").is_ok() {
            installed = true;
            evidence.push("zcode binary found in PATH".to_string());
        }

        if let Some(env_path) = Self::env_db_path() {
            evidence.push(format!("env override target: {}", env_path.display()));
        }

        let dbs = Self::find_db_files();
        if !dbs.is_empty() {
            installed = true;
            evidence.push(format!("found {} zcode.db database(s)", dbs.len()));
        }

        trace!(provider = "zcode", installed, ?evidence, "detection");
        DetectionResult {
            installed,
            version: None,
            evidence,
        }
    }

    fn session_roots(&self) -> Vec<PathBuf> {
        Self::find_db_files()
    }

    fn owns_session(&self, session_id: &str) -> Option<PathBuf> {
        // Try both with and without zc_ prefix.
        let candidates: Vec<String> = if session_id.starts_with("zc_") {
            vec![session_id.to_string()]
        } else {
            vec![session_id.to_string(), format!("zc_{session_id}")]
        };

        for db_path in Self::find_db_files() {
            let Ok(conn) = Self::open_db(&db_path) else {
                continue;
            };
            for id in &candidates {
                if Self::session_exists(&conn, id) {
                    let virtual_path = Self::virtual_session_path(&db_path, id);
                    debug!(
                        db = %db_path.display(),
                        session = %virtual_path.display(),
                        session_id = %id,
                        "found ZCode session"
                    );
                    return Some(virtual_path);
                }
            }
        }
        None
    }

    fn read_session(&self, path: &Path) -> anyhow::Result<CanonicalSession> {
        debug!(path = %path.display(), "reading ZCode session");

        // Virtual path (`.../db.sqlite/<encoded-session-id>`) from discovery.
        if let Some((db_path, session_id)) = Self::parse_virtual_path(path) {
            let conn = Self::open_db(&db_path)?;
            return Self::read_session_by_id(&conn, &db_path, &session_id);
        }

        // Direct DB path — choose newest root session.
        let conn = Self::open_db(path)?;
        let Some(session_id) = Self::newest_root_session_id(&conn) else {
            anyhow::bail!("no ZCode sessions found in {}", path.display());
        };
        Self::read_session_by_id(&conn, path, &session_id)
    }

    fn write_session(
        &self,
        session: &CanonicalSession,
        opts: &WriteOptions,
    ) -> anyhow::Result<WrittenSession> {
        let db_path = Self::choose_target_db_path()?;
        let mut conn = Self::open_db_rw(&db_path)?;

        // Prefer pipeline-supplied deterministic id, else source session id,
        // else a fresh UUID. ZCode uses plain UUIDs (no prefix required),
        // but we add zc_ for disambiguation.
        let raw_id = opts
            .target_session_id
            .clone()
            .filter(|s| !s.trim().is_empty())
            .or_else(|| {
                if session.session_id.is_empty() {
                    None
                } else {
                    Some(session.session_id.clone())
                }
            })
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let target_session_id = Self::ensure_zc_prefix(&raw_id);

        // Detect or create schema.
        let schema =
            if Self::table_exists(&conn, "session") || Self::table_exists(&conn, "sessions") {
                Self::detect_schema(&conn)
            } else {
                // Create V2 schema for fresh DBs.
                ensure_zcode_schema(&conn)?;
                SchemaKind::V2
            };

        if Self::session_exists(&conn, &target_session_id) {
            if opts.force {
                delete_session_cascade(&conn, schema, &target_session_id)?;
            } else {
                return Err(crate::error::CasrError::SessionConflict {
                    session_id: target_session_id,
                    existing_path: db_path,
                }
                .into());
            }
        }

        match schema {
            SchemaKind::V2 => write_session_v2(&mut conn, session, &target_session_id)?,
            SchemaKind::Legacy => {
                write_session_legacy(&mut conn, session, &target_session_id)?;
            }
        }

        let virtual_path = Self::virtual_session_path(&db_path, &target_session_id);
        info!(
            session_id = target_session_id,
            path = %db_path.display(),
            schema = ?schema,
            messages = session.messages.len(),
            "ZCode session written"
        );

        Ok(WrittenSession {
            paths: vec![virtual_path],
            session_id: target_session_id.clone(),
            resume_command: self.resume_command(&target_session_id),
            backup_path: None,
            warnings: Vec::new(),
        })
    }

    fn resume_command(&self, session_id: &str) -> String {
        let id = Self::ensure_zc_prefix(session_id);
        format!("zcode --resume {id}")
    }

    fn list_sessions(&self) -> Option<Vec<(String, PathBuf)>> {
        let db_files = Self::find_db_files();
        if db_files.is_empty() {
            return Some(Vec::new());
        }

        let mut results = Vec::new();
        for db_path in &db_files {
            let Ok(conn) = Self::open_db(db_path) else {
                continue;
            };
            let sql = match Self::detect_schema(&conn) {
                SchemaKind::V2 => {
                    "SELECT id FROM session WHERE task_type = 'interactive'
                     ORDER BY time_created DESC"
                }
                SchemaKind::Legacy => {
                    if !Self::table_exists(&conn, "sessions") {
                        continue;
                    }
                    "SELECT id FROM sessions ORDER BY created_at DESC"
                }
            };

            let Ok(mut stmt) = conn.prepare(sql) else {
                continue;
            };

            let Ok(rows) = stmt.query_map([], |row| row.get::<_, String>(0)) else {
                continue;
            };

            for row in rows.flatten() {
                let virtual_path = Self::virtual_session_path(db_path, &row);
                results.push((row, virtual_path));
            }
        }

        Some(results)
    }
}

// ---------------------------------------------------------------------------
// Schema & write helpers (top-level functions to avoid clippy issues)
// ---------------------------------------------------------------------------

/// On-disk schema flavor for a ZCode database.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SchemaKind {
    /// ZCode V2: `session` / `message` / `part`.
    V2,
    /// Legacy layout: `sessions` / `messages`.
    Legacy,
}

fn ensure_zcode_schema(conn: &Connection) -> anyhow::Result<()> {
    conn.execute_batch(
        r#"
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS session (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL DEFAULT 'global',
    workspace_id TEXT,
    parent_id TEXT,
    slug TEXT NOT NULL DEFAULT '',
    directory TEXT NOT NULL DEFAULT '',
    path TEXT,
    title TEXT NOT NULL DEFAULT '',
    version TEXT NOT NULL DEFAULT '1.0',
    share_url TEXT,
    summary_additions INTEGER,
    summary_deletions INTEGER,
    summary_files INTEGER,
    summary_diffs TEXT,
    revert TEXT,
    permission TEXT,
    time_created INTEGER NOT NULL,
    time_updated INTEGER NOT NULL,
    time_compacting INTEGER,
    time_archived INTEGER,
    task_type TEXT NOT NULL DEFAULT 'interactive',
    title_source TEXT NOT NULL DEFAULT 'first_input',
    title_message_id TEXT,
    time_title_updated INTEGER,
    trace_id TEXT
);

CREATE TABLE IF NOT EXISTS message (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    time_created INTEGER NOT NULL,
    time_updated INTEGER NOT NULL,
    data TEXT NOT NULL DEFAULT '{}',
    sequence INTEGER
);

CREATE TABLE IF NOT EXISTS part (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL REFERENCES message(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    time_created INTEGER NOT NULL,
    time_updated INTEGER NOT NULL,
    data TEXT NOT NULL DEFAULT '{}',
    sequence INTEGER
);

CREATE INDEX IF NOT EXISTS idx_zcode_message_session ON message (session_id, time_created, id);
CREATE INDEX IF NOT EXISTS idx_zcode_part_message ON part (message_id, id);

CREATE TRIGGER IF NOT EXISTS message_sequence_autofill
    AFTER INSERT ON message
    WHEN NEW.sequence IS NULL
    BEGIN
        UPDATE message
        SET sequence = (
            SELECT COALESCE(MAX(sequence), -1) + 1
            FROM message
            WHERE session_id = NEW.session_id
        )
        WHERE id = NEW.id;
    END;

CREATE TRIGGER IF NOT EXISTS part_sequence_autofill
    AFTER INSERT ON part
    WHEN NEW.sequence IS NULL
    BEGIN
        UPDATE part
        SET sequence = (
            SELECT COALESCE(MAX(sequence), -1) + 1
            FROM part
            WHERE message_id = NEW.message_id
        )
        WHERE id = NEW.id;
    END;
"#,
    )
    .context("failed to initialize ZCode schema")?;
    Ok(())
}

fn delete_session_cascade(
    conn: &Connection,
    schema: SchemaKind,
    session_id: &str,
) -> anyhow::Result<()> {
    match schema {
        SchemaKind::V2 => {
            for table in ["part", "message"] {
                let sql = format!("DELETE FROM {table} WHERE session_id = ?1");
                let _ = conn.execute(&sql, rusqlite::params![session_id]);
            }
            conn.execute(
                "DELETE FROM session WHERE id = ?1",
                rusqlite::params![session_id],
            )
            .context("failed to delete existing ZCode v2 session for --force")?;
        }
        SchemaKind::Legacy => {
            let _ = conn.execute(
                "DELETE FROM messages WHERE session_id = ?1",
                rusqlite::params![session_id],
            );
            conn.execute(
                "DELETE FROM sessions WHERE id = ?1",
                rusqlite::params![session_id],
            )
            .context("failed to delete existing ZCode legacy session for --force")?;
        }
    }
    Ok(())
}

fn write_session_legacy(
    conn: &mut Connection,
    session: &CanonicalSession,
    target_session_id: &str,
) -> anyhow::Result<()> {
    let now = chrono::Utc::now().timestamp_millis();
    let created_at = session.started_at.unwrap_or(now);
    let updated_at = session.ended_at.unwrap_or(now);
    let title = session_title(session);

    let tx = conn.transaction().context("failed to begin transaction")?;

    tx.execute(
        "INSERT INTO sessions (
            id, parent_session_id, title, message_count, prompt_tokens, completion_tokens, cost,
            summary_message_id, updated_at, created_at
         ) VALUES (?1, NULL, ?2, ?3, 0, 0, 0.0, NULL, ?4, ?5)",
        rusqlite::params![
            target_session_id,
            title,
            i64::try_from(session.messages.len()).unwrap_or(i64::MAX),
            updated_at,
            created_at,
        ],
    )
    .context("failed to insert ZCode legacy session")?;

    let default_model = session.model_name.clone();
    for msg in &session.messages {
        let message_id = uuid::Uuid::new_v4().to_string();
        let parts = build_parts(msg);
        let parts_json =
            serde_json::to_string(&parts).context("failed to serialize ZCode parts")?;
        let timestamp = msg.timestamp.unwrap_or(created_at);
        let model = msg.author.clone().or_else(|| default_model.clone());

        tx.execute(
            "INSERT INTO messages (
                id, session_id, role, parts, model, created_at, updated_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                message_id,
                target_session_id,
                role_to_zcode(&msg.role),
                parts_json,
                model,
                timestamp,
                timestamp,
            ],
        )
        .with_context(|| format!("failed to insert ZCode message {}", msg.idx))?;
    }

    tx.commit().context("failed to commit legacy write")?;
    Ok(())
}

fn write_session_v2(
    conn: &mut Connection,
    session: &CanonicalSession,
    target_session_id: &str,
) -> anyhow::Result<()> {
    let now = chrono::Utc::now().timestamp_millis();
    let created_at = session.started_at.unwrap_or(now);
    let updated_at = session.ended_at.unwrap_or(now);
    let title = session_title(session);

    let directory = session
        .workspace
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    let path_field = directory.trim_start_matches('/').to_string();

    let slug = slug_from_title(&title);

    let tx = conn
        .transaction()
        .context("failed to begin v2 transaction")?;

    tx.execute(
        "INSERT INTO session (
            id, project_id, workspace_id, parent_id, slug, directory, path, title, version,
            share_url, summary_additions, summary_deletions, summary_files, summary_diffs,
            revert, permission,
            time_created, time_updated, time_compacting, time_archived,
            task_type, title_source, title_message_id, time_title_updated, trace_id
         ) VALUES (
            ?1, 'global', NULL, NULL, ?2, ?3, ?4, ?5, '1.0',
            NULL, NULL, NULL, NULL, NULL,
            NULL, NULL,
            ?6, ?7, NULL, NULL,
            'interactive', 'custom', NULL, NULL, NULL
         )",
        rusqlite::params![
            target_session_id,
            slug,
            directory,
            path_field,
            title,
            created_at,
            updated_at,
        ],
    )
    .context("failed to insert ZCode v2 session")?;

    let mut parent_msg_id: Option<String> = None;
    for msg in &session.messages {
        let message_id = ZCode::mint_entity_id("msg_");
        let timestamp = msg.timestamp.unwrap_or(created_at);
        let model = msg
            .author
            .clone()
            .or_else(|| session.model_name.clone())
            .unwrap_or_else(|| "unknown".to_string());

        let mut data = serde_json::json!({
            "role": role_to_zcode(&msg.role),
            "time": { "created": timestamp },
            "agent": "build",
        });

        match msg.role {
            MessageRole::User => {
                data["model"] = serde_json::json!({
                    "providerID": "zcode",
                    "modelID": model,
                });
            }
            MessageRole::Assistant => {
                if let Some(parent) = &parent_msg_id {
                    data["parentID"] = serde_json::Value::String(parent.clone());
                }
                data["mode"] = serde_json::json!("build");
                data["modelID"] = serde_json::Value::String(model.clone());
                data["providerID"] = serde_json::json!("zcode");
                data["time"] = serde_json::json!({
                    "created": timestamp,
                    "completed": timestamp,
                });
                data["finish"] = serde_json::json!("stop");
            }
            _ => {}
        }

        let data_json =
            serde_json::to_string(&data).context("serialize ZCode v2 message data")?;

        tx.execute(
            "INSERT INTO message (id, session_id, time_created, time_updated, data)
             VALUES (?1, ?2, ?3, ?3, ?4)",
            rusqlite::params![message_id, target_session_id, timestamp, data_json],
        )
        .with_context(|| format!("failed to insert ZCode v2 message {}", msg.idx))?;

        // Emit flat part rows.
        let part_specs = build_parts_v2(msg);
        for part_data in part_specs {
            let part_id = ZCode::mint_entity_id("prt_");
            let mut pdata = part_data;
            if let Some(obj) = pdata.as_object_mut() {
                obj.insert("id".to_string(), serde_json::Value::String(part_id.clone()));
                obj.insert(
                    "sessionID".to_string(),
                    serde_json::Value::String(target_session_id.to_string()),
                );
                obj.insert(
                    "messageID".to_string(),
                    serde_json::Value::String(message_id.clone()),
                );
            }
            let pjson = serde_json::to_string(&pdata).context("serialize ZCode v2 part")?;
            tx.execute(
                "INSERT INTO part (id, message_id, session_id, time_created, time_updated, data)
                 VALUES (?1, ?2, ?3, ?4, ?4, ?5)",
                rusqlite::params![part_id, message_id, target_session_id, timestamp, pjson],
            )
            .context("failed to insert ZCode v2 part")?;
        }

        if matches!(msg.role, MessageRole::User | MessageRole::Assistant) {
            parent_msg_id = Some(message_id);
        }
    }

    tx.commit().context("failed to commit v2 write")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared helper functions
// ---------------------------------------------------------------------------

fn dedup_existing_files(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen: BTreeSet<PathBuf> = BTreeSet::new();
    for path in paths {
        if path.is_file() {
            seen.insert(path);
        }
    }
    seen.into_iter().collect()
}

fn session_title(session: &CanonicalSession) -> String {
    session
        .title
        .clone()
        .or_else(|| {
            session
                .messages
                .iter()
                .find(|m| m.role == MessageRole::User)
                .map(|m| truncate_title(&m.content, 80))
                .filter(|t| !t.is_empty())
        })
        .unwrap_or_else(|| "Converted session".to_string())
}

fn slug_from_title(title: &str) -> String {
    let mut slug: String = title
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    while slug.contains("--") {
        slug = slug.replace("--", "-");
    }
    let slug = slug.trim_matches('-');
    if slug.is_empty() {
        "casr-import".to_string()
    } else {
        slug.chars().take(40).collect()
    }
}

fn parse_tool_call_arguments(input: &str) -> serde_json::Value {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return serde_json::json!({});
    }
    serde_json::from_str(trimmed).unwrap_or_else(|_| serde_json::json!({ "input": input }))
}

fn parse_parts(parts: &serde_json::Value) -> (String, Vec<ToolCall>, Vec<ToolResult>) {
    let mut text_chunks: Vec<String> = Vec::new();
    let mut reasoning_chunks: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut tool_results: Vec<ToolResult> = Vec::new();

    let Some(items) = parts.as_array() else {
        return (String::new(), tool_calls, tool_results);
    };

    for item in items {
        let part_type = item
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        let nested = item.get("data");
        let data = nested.unwrap_or(item);

        match part_type {
            "text" => {
                let text = data
                    .get("text")
                    .and_then(serde_json::Value::as_str)
                    .or_else(|| item.get("text").and_then(serde_json::Value::as_str));
                if let Some(text) = text
                    && !text.trim().is_empty()
                {
                    text_chunks.push(text.to_string());
                }
            }
            "reasoning" => {
                let thinking = data
                    .get("thinking")
                    .or_else(|| data.get("text"))
                    .and_then(serde_json::Value::as_str)
                    .or_else(|| item.get("text").and_then(serde_json::Value::as_str));
                if let Some(thinking) = thinking
                    && !thinking.trim().is_empty()
                {
                    reasoning_chunks.push(thinking.to_string());
                }
            }
            "tool_call" | "tool" => {
                let name = data
                    .get("name")
                    .or_else(|| data.get("tool"))
                    .or_else(|| item.get("tool"))
                    .and_then(serde_json::Value::as_str)
                    .filter(|name| !name.is_empty())
                    .unwrap_or("tool_call")
                    .to_string();
                let id = data
                    .get("id")
                    .or_else(|| data.get("callID"))
                    .or_else(|| item.get("callID"))
                    .and_then(serde_json::Value::as_str)
                    .filter(|id| !id.is_empty())
                    .map(ToString::to_string);
                let input = data
                    .get("input")
                    .or_else(|| data.pointer("/state/input"))
                    .map(|v| {
                        if let Some(s) = v.as_str() {
                            s.to_string()
                        } else {
                            v.to_string()
                        }
                    })
                    .unwrap_or_default();

                tool_calls.push(ToolCall {
                    id,
                    name,
                    arguments: parse_tool_call_arguments(&input),
                });
            }
            "tool_result" => {
                let content = data
                    .get("content")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let call_id = data
                    .get("tool_call_id")
                    .or_else(|| data.get("callID"))
                    .and_then(serde_json::Value::as_str)
                    .filter(|id| !id.is_empty())
                    .map(ToString::to_string);
                let is_error = data
                    .get("is_error")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);

                tool_results.push(ToolResult {
                    call_id,
                    content,
                    is_error,
                });
            }
            "step-start" | "step-finish" => {
                // Structural markers — ignore.
            }
            "compaction" => {
                // ZCode compaction markers — skip silently.
            }
            _ => {
                let fallback = flatten_content(data);
                if !fallback.trim().is_empty() {
                    text_chunks.push(fallback);
                }
            }
        }
    }

    let mut content = text_chunks.join("\n");
    if content.trim().is_empty() {
        content = reasoning_chunks.join("\n");
    }
    if content.trim().is_empty() {
        let result_texts: Vec<&str> = tool_results
            .iter()
            .map(|result| result.content.as_str())
            .filter(|text| !text.trim().is_empty())
            .collect();
        content = result_texts.join("\n");
    }

    (content, tool_calls, tool_results)
}

fn build_parts(message: &CanonicalMessage) -> serde_json::Value {
    let mut parts = Vec::new();

    if !message.content.trim().is_empty() {
        parts.push(serde_json::json!({
            "type": "text",
            "data": { "text": message.content },
        }));
    }

    for call in &message.tool_calls {
        let input = if let Some(s) = call.arguments.as_str() {
            s.to_string()
        } else {
            serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_string())
        };

        parts.push(serde_json::json!({
            "type": "tool_call",
            "data": {
                "id": call.id.clone().unwrap_or_default(),
                "name": call.name,
                "input": input,
                "type": "function",
                "finished": true
            }
        }));
    }

    for result in &message.tool_results {
        parts.push(serde_json::json!({
            "type": "tool_result",
            "data": {
                "tool_call_id": result.call_id.clone().unwrap_or_default(),
                "name": "tool",
                "content": result.content,
                "metadata": "",
                "is_error": result.is_error
            }
        }));
    }

    serde_json::Value::Array(parts)
}

/// V2 part rows store flat objects (`{"type":"text","text":"..."}`).
///
/// Tool results are embedded as `state.output` in the tool part. We avoid
/// emitting separate text parts for tool results when the message already has
/// content — doing so would cause the read-back content to include the tool
/// output a second time, breaking round-trip fidelity.
fn build_parts_v2(message: &CanonicalMessage) -> Vec<serde_json::Value> {
    let mut parts = Vec::new();

    if !message.content.trim().is_empty() {
        parts.push(serde_json::json!({
            "type": "text",
            "text": message.content,
        }));
    }

    for call in &message.tool_calls {
        let input = if let Some(s) = call.arguments.as_str() {
            serde_json::from_str::<serde_json::Value>(s).unwrap_or_else(|_| serde_json::json!({}))
        } else if call.arguments.is_null() {
            serde_json::json!({})
        } else {
            call.arguments.clone()
        };
        let output = message
            .tool_results
            .iter()
            .find(|tr| tr.call_id.as_deref() == call.id.as_deref())
            .map(|tr| tr.content.clone())
            .unwrap_or_default();
        let ts = message.timestamp.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0)
        });
        parts.push(serde_json::json!({
            "type": "tool",
            "tool": call.name,
            "callID": call.id.clone().unwrap_or_default(),
            "state": {
                "status": "completed",
                "input": input,
                "output": output,
                "metadata": {},
                "title": "",
                "time": {"start": ts, "end": ts},
            }
        }));
    }

    // Tool results are already captured in the tool part's state.output.
    // Emitting them as separate text parts would duplicate content on read-back
    // when the message already has text. Only emit for orphan results (no
    // matching tool call) that carry meaningful content.
    if !message.content.trim().is_empty() {
        // Main content already present — skip tool result text parts.
    } else {
        for result in &message.tool_results {
            if !result.content.trim().is_empty() {
                parts.push(serde_json::json!({
                    "type": "text",
                    "text": result.content,
                }));
            }
        }
    }

    if parts.is_empty() {
        parts.push(serde_json::json!({
            "type": "text",
            "text": "",
        }));
    }

    parts
}

fn role_to_zcode(role: &MessageRole) -> &str {
    match role {
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
        MessageRole::System => "system",
        MessageRole::Other(role) => role.as_str(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::Provider;
    use std::sync::{LazyLock, Mutex};

    static ZCODE_ENV: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    struct EnvDbGuard;

    impl EnvDbGuard {
        fn pin(db_path: &Path) -> Self {
            TEST_DB_PATH_OVERRIDE.with(|cell| {
                *cell.borrow_mut() = Some(db_path.to_path_buf());
            });
            Self
        }
    }

    impl Drop for EnvDbGuard {
        fn drop(&mut self) {
            TEST_DB_PATH_OVERRIDE.with(|cell| {
                *cell.borrow_mut() = None;
            });
        }
    }

    /// Temp workspace + isolated ZCode DB for write tests.
    fn test_workspace() -> (tempfile::TempDir, PathBuf, EnvDbGuard) {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let db_dir = tmp.path().join(".zcode").join("cli").join("db");
        std::fs::create_dir_all(&db_dir).expect("db dir");
        let db = db_dir.join("db.sqlite");
        let env = EnvDbGuard::pin(&db);
        (tmp, db, env)
    }

    fn sample_session(workspace: &Path) -> CanonicalSession {
        CanonicalSession {
            session_id: "source-session".to_string(),
            provider_slug: "claude-code".to_string(),
            workspace: Some(workspace.to_path_buf()),
            title: Some("Fix ZCode adapter".to_string()),
            started_at: Some(1_700_000_000_000),
            ended_at: Some(1_700_000_010_000),
            messages: vec![
                CanonicalMessage {
                    idx: 0,
                    role: MessageRole::User,
                    content: "Please inspect src/main.rs".to_string(),
                    timestamp: Some(1_700_000_000_000),
                    author: None,
                    tool_calls: vec![],
                    tool_results: vec![],
                    extra: serde_json::json!({}),
                },
                CanonicalMessage {
                    idx: 1,
                    role: MessageRole::Assistant,
                    content: "Inspecting now.".to_string(),
                    timestamp: Some(1_700_000_005_000),
                    author: Some("GLM-5.3".to_string()),
                    tool_calls: vec![ToolCall {
                        id: Some("call-1".to_string()),
                        name: "Read".to_string(),
                        arguments: serde_json::json!({"path":"src/main.rs"}),
                    }],
                    tool_results: vec![ToolResult {
                        call_id: Some("call-1".to_string()),
                        content: "Read complete".to_string(),
                        is_error: false,
                    }],
                    extra: serde_json::json!({}),
                },
            ],
            metadata: serde_json::json!({}),
            source_path: workspace.join("source.jsonl"),
            model_name: Some("GLM-5.3".to_string()),
        }
    }

    #[test]
    fn provider_metadata_and_resume_command() {
        let provider = ZCode;
        assert_eq!(provider.name(), "ZCode");
        assert_eq!(provider.slug(), "zcode");
        assert_eq!(provider.cli_alias(), "zc");
        assert_eq!(
            <ZCode as Provider>::resume_command(&provider, "sid"),
            "zcode --resume zc_sid"
        );
        assert_eq!(
            <ZCode as Provider>::resume_command(&provider, "zc_already"),
            "zcode --resume zc_already"
        );
    }

    #[test]
    fn virtual_path_round_trip() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let db = tmp.path().join("db.sqlite");
        std::fs::write(&db, "").expect("touch db file");

        let sid = "abc-123";
        let virtual_path = ZCode::virtual_session_path(&db, sid);
        let parsed = ZCode::parse_virtual_path(&virtual_path).expect("should parse");
        assert_eq!(parsed.0, db.as_path());
        assert_eq!(parsed.1, sid);
    }

    #[test]
    fn writer_reader_roundtrip_preserves_core_content() {
        let _lock = ZCODE_ENV.lock().expect("mutex lock");
        let (_tmp, _db_path, _env) = test_workspace();
        let workspace = _tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).expect("workspace dir");

        let source = sample_session(&workspace);
        let written = ZCode
            .write_session(
                &source,
                &WriteOptions {
                    force: false,
                    target_session_id: None,
                },
            )
            .expect("write should succeed");

        assert!(
            written.resume_command.starts_with("zcode --resume zc_"),
            "resume_command={}",
            written.resume_command
        );
        assert_eq!(written.paths.len(), 1);
        let db_path = written
            .paths
            .first()
            .and_then(|p| p.parent())
            .expect("virtual path parent");
        assert!(db_path.is_file(), "db file should exist");

        let readback = ZCode
            .read_session(&written.paths[0])
            .expect("readback should succeed");

        assert_eq!(readback.provider_slug, "zcode");
        assert_eq!(readback.messages.len(), source.messages.len());
        assert_eq!(readback.messages[0].role, MessageRole::User);
        assert_eq!(readback.messages[0].content, source.messages[0].content);
        assert_eq!(readback.messages[1].role, MessageRole::Assistant);
        assert_eq!(readback.messages[1].content, source.messages[1].content);
        assert_eq!(readback.workspace.as_deref(), Some(workspace.as_path()));
        // Target id is source id with required `zc_` prefix.
        assert_eq!(readback.session_id, format!("zc_{}", source.session_id));
    }

    #[test]
    fn write_twice_with_force_overwrites_in_place() {
        let _lock = ZCODE_ENV.lock().expect("mutex lock");
        let (_tmp, _db_path, _env) = test_workspace();
        let workspace = _tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).expect("workspace dir");

        let source = sample_session(&workspace);

        // First write succeeds.
        let first = ZCode
            .write_session(
                &source,
                &WriteOptions {
                    force: false,
                    target_session_id: None,
                },
            )
            .expect("first write should succeed");
        let db_path = first.paths[0].parent().expect("db parent").to_path_buf();

        // Second write WITHOUT force must be a clean conflict.
        let conflict = ZCode
            .write_session(
                &source,
                &WriteOptions {
                    force: false,
                    target_session_id: None,
                },
            )
            .expect_err("second write without --force should conflict");
        let expected_id = format!("zc_{}", source.session_id);
        match conflict.downcast_ref::<crate::error::CasrError>() {
            Some(crate::error::CasrError::SessionConflict { session_id, .. }) => {
                assert_eq!(session_id, &expected_id);
            }
            other => panic!("expected SessionConflict, got {other:?}"),
        }

        // Second write WITH force succeeds and overwrites in place.
        let second = ZCode
            .write_session(
                &source,
                &WriteOptions {
                    force: true,
                    target_session_id: None,
                },
            )
            .expect("force write should succeed");

        assert_eq!(first.session_id, second.session_id);
        assert_eq!(second.session_id, expected_id);

        // Exactly one session row.
        let conn = ZCode::open_db(&db_path).expect("open db");
        let session_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM session", [], |r| r.get(0))
            .expect("count sessions");
        assert_eq!(session_count, 1, "force must overwrite, not duplicate");

        let message_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM message", [], |r| r.get(0))
            .expect("count messages");
        assert_eq!(
            message_count,
            source.messages.len() as i64,
            "messages from the prior write must be replaced, not accumulated"
        );

        // The overwritten session still reads back cleanly.
        let readback = ZCode
            .read_session(&second.paths[0])
            .expect("readback after force overwrite");
        assert_eq!(readback.messages.len(), source.messages.len());
    }

    #[test]
    fn owns_session_returns_virtual_path() {
        let _lock = ZCODE_ENV.lock().expect("mutex lock");
        let (_tmp, _db_path, _env) = test_workspace();
        let workspace = _tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).expect("workspace dir");

        let source = sample_session(&workspace);
        let written = ZCode
            .write_session(
                &source,
                &WriteOptions {
                    force: false,
                    target_session_id: None,
                },
            )
            .expect("write should succeed");
        let found = ZCode.owns_session(&written.session_id);
        assert!(found.is_some(), "owns_session should find written id");
        assert!(
            written.session_id.starts_with("zc_"),
            "ZCode target ids must use zc_ prefix"
        );
    }

    #[test]
    fn list_sessions_returns_written_session() {
        let _lock = ZCODE_ENV.lock().expect("mutex lock");
        let (_tmp, _db_path, _env) = test_workspace();
        let workspace = _tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).expect("workspace dir");

        let source = sample_session(&workspace);
        let written = ZCode
            .write_session(
                &source,
                &WriteOptions {
                    force: false,
                    target_session_id: None,
                },
            )
            .expect("write");

        let listed = ZCode.list_sessions().expect("should return Some");
        let ids: Vec<&str> = listed.iter().map(|(id, _)| id.as_str()).collect();
        assert!(
            ids.contains(&written.session_id.as_str()),
            "written session should be listed, found: {ids:?}"
        );
    }

    #[test]
    fn parse_parts_extracts_tool_calls_and_results() {
        let raw = serde_json::json!([
            {"type":"text","data":{"text":"hello"}},
            {"type":"tool_call","data":{"id":"c1","name":"Read","input":"{\"path\":\"src/main.rs\"}"}},
            {"type":"tool_result","data":{"tool_call_id":"c1","content":"ok","is_error":false}}
        ]);

        let (content, tool_calls, tool_results) = parse_parts(&raw);
        assert_eq!(content, "hello");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "Read");
        assert_eq!(tool_results.len(), 1);
        assert_eq!(tool_results[0].content, "ok");
    }

    #[test]
    fn parse_parts_skips_compaction_markers() {
        let raw = serde_json::json!([
            {"type":"text","data":{"text":"before"}},
            {"type":"compaction","summaryMessageId":"s1","compactBoundary":true},
            {"type":"text","data":{"text":"after"}}
        ]);
        let (content, _, _) = parse_parts(&raw);
        assert_eq!(content, "before\nafter");
    }

    #[test]
    fn role_to_zcode_all_variants() {
        assert_eq!(role_to_zcode(&MessageRole::User), "user");
        assert_eq!(role_to_zcode(&MessageRole::Assistant), "assistant");
        assert_eq!(role_to_zcode(&MessageRole::Tool), "tool");
        assert_eq!(role_to_zcode(&MessageRole::System), "system");
        assert_eq!(
            role_to_zcode(&MessageRole::Other("custom".to_string())),
            "custom"
        );
    }
}

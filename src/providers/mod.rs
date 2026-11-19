//! Provider trait and concrete provider implementations.
//!
//! Each supported provider (Claude Code, Codex, Gemini CLI, Antigravity CLI,
//! Cursor, Cline, Aider, Amp, OpenCode, ChatGPT, ClawdBot, Vibe, Factory,
//! OpenClaw, Pi-Agent, OMP (oh-my-pi), Kiro, Grok, Hermes, jcode) implements
//! the [`Provider`] trait to read/write sessions in its native format.

pub mod aider;
pub mod amp;
pub mod antigravity;
pub mod chatgpt;
pub mod claude_code;
pub mod clawdbot;
pub mod cline;
pub mod codex;
pub mod cursor;
pub mod factory;
pub mod gemini;
pub mod grok;
pub mod hermes;
pub mod jcode;
pub mod kiro;
pub mod openclaw;
pub mod opencode;
pub mod pi_agent;
pub mod vibe;
pub mod zcode;

use std::path::{Path, PathBuf};

use crate::discovery::DetectionResult;
use crate::model::CanonicalSession;

/// Options controlling how a session is written to disk.
#[derive(Debug, Clone, Default)]
pub struct WriteOptions {
    /// Overwrite existing session file (creates `.bak` backup).
    pub force: bool,
    /// Use this id as the target session id (and filename stem). When `None`,
    /// the writer is free to pick its own id (historically a fresh UUID).
    /// The pipeline supplies a deterministic id derived from
    /// `(source_provider_alias, source_session_id)` so re-running the same
    /// conversion never produces duplicates.
    pub target_session_id: Option<String>,
}

/// Resolve the session id a file-backed writer should use.
///
/// Prefers the pipeline-supplied deterministic [`WriteOptions::target_session_id`]
/// (which makes re-running the same conversion idempotent), then falls back to
/// the source session id, and finally to a timestamped `casr-` id when neither
/// is usable. Every candidate passes through
/// [`crate::discovery::sanitize_session_id_for_path`] so untrusted ids from
/// source files cannot escape the provider's session root via `..`, separators,
/// or NUL bytes.
pub(crate) fn resolve_write_session_id(opts: &WriteOptions, session: &CanonicalSession) -> String {
    if let Some(id) = opts
        .target_session_id
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return crate::discovery::sanitize_session_id_for_path(id);
    }
    if !session.session_id.trim().is_empty() {
        return crate::discovery::sanitize_session_id_for_path(&session.session_id);
    }
    format!("casr-{}", chrono::Utc::now().format("%Y%m%dT%H%M%S"))
}

/// Quote a session id for safe interpolation into a POSIX shell command line.
///
/// Ids whose first char is `[A-Za-z0-9]` and whose remaining chars are in
/// `[A-Za-z0-9._/:@,+%-]` pass through unquoted. Anything else is wrapped in
/// single quotes with embedded `'` escaped as `'\''`. A leading `-` is always
/// quoted so the value cannot be reparsed as a CLI flag.
pub(crate) fn shell_quote_arg(arg: &str) -> String {
    let safe = !arg.is_empty()
        && arg.chars().enumerate().all(|(i, c)| {
            if i == 0 {
                c.is_ascii_alphanumeric()
            } else {
                c.is_ascii_alphanumeric()
                    || matches!(c, '.' | '_' | '/' | ':' | '@' | ',' | '+' | '%' | '-')
            }
        });
    if safe {
        return arg.to_string();
    }
    format!("'{}'", arg.replace('\'', "'\\''"))
}

/// Escape a string for interpolation inside a double-quoted shell argument.
pub(crate) fn shell_escape_dq(arg: &str) -> String {
    let mut out = String::with_capacity(arg.len());
    for c in arg.chars() {
        if matches!(c, '"' | '\\' | '$' | '`') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Describes the files produced by a successful write operation.
#[derive(Debug, Clone)]
pub struct WrittenSession {
    /// Paths of files written.
    pub paths: Vec<PathBuf>,
    /// Session ID in the target provider's format.
    pub session_id: String,
    /// Ready-to-paste command to resume the session.
    pub resume_command: String,
    /// Path to the `.bak` backup, if an existing file was overwritten.
    pub backup_path: Option<PathBuf>,
    /// Non-fatal warnings produced while writing (e.g. the target session was
    /// written but could not be registered in the provider's resume index).
    /// Surfaced to the user and merged into the conversion's warning list.
    pub warnings: Vec<String>,
}

/// The core abstraction each provider implements.
///
/// Object-safe so we can store `Box<dyn Provider>` in the registry.
pub trait Provider: Send + Sync {
    /// Human-readable name (e.g. `"Claude Code"`).
    fn name(&self) -> &str;

    /// Short slug used in session metadata (e.g. `"claude-code"`).
    fn slug(&self) -> &str;

    /// CLI alias used in `casr <alias> resume …` (e.g. `"cc"`).
    fn cli_alias(&self) -> &str;

    /// Probe whether this provider is installed on the machine.
    fn detect(&self) -> DetectionResult;

    /// Root directories where this provider stores sessions.
    fn session_roots(&self) -> Vec<PathBuf>;

    /// Check if `session_id` belongs to this provider; return the file path if so.
    fn owns_session(&self, session_id: &str) -> Option<PathBuf>;

    /// Read a session from its native format into canonical IR.
    fn read_session(&self, path: &Path) -> anyhow::Result<CanonicalSession>;

    /// Read only the trailing `tail_messages` of a session, for preview use
    /// (`casr info --peek`). The default performs a full [`Provider::read_session`]
    /// and drops the leading messages; JSONL providers may override with a
    /// bounded reverse read. Session-level fields (id, workspace, timestamps,
    /// metadata) are still populated.
    fn read_session_tail(
        &self,
        path: &Path,
        tail_messages: usize,
    ) -> anyhow::Result<CanonicalSession> {
        let mut session = self.read_session(path)?;
        if session.messages.len() > tail_messages {
            let excess = session.messages.len() - tail_messages;
            session.messages.drain(..excess);
        }
        Ok(session)
    }

    /// Write a canonical session into this provider's native format.
    fn write_session(
        &self,
        session: &CanonicalSession,
        opts: &WriteOptions,
    ) -> anyhow::Result<WrittenSession>;

    /// Whether this provider can act as a conversion *target*.
    ///
    /// Read/resume-only providers (e.g. Antigravity) return `false`; their
    /// [`Provider::write_session`] refuses to synthesize a resumable session.
    /// Surfaced in `providers --json` as `target_capable`.
    fn target_capable(&self) -> bool {
        true
    }

    /// Build the shell command to resume a session with this provider.
    fn resume_command(&self, session_id: &str) -> String;

    /// Enumerate all discoverable sessions for this provider.
    ///
    /// Returns `Some(vec)` of `(session_id, path)` pairs when the provider
    /// stores multiple sessions in a single file or database and directory
    /// walking alone would undercount.  The default returns `None`, which
    /// tells the caller to fall back to directory walking + `read_session`.
    fn list_sessions(&self) -> Option<Vec<(String, PathBuf)>> {
        None
    }

    /// Remove persistent state produced by [`Provider::write_session`] after a
    /// failed read-back verification.
    ///
    /// File-based providers are rolled back by the pipeline via path deletion
    /// (and `.bak` restore); DB-backed providers override this to delete or
    /// restore their rows since their `WrittenSession::paths` are virtual and
    /// cannot be removed as files. The default is a no-op.
    fn rollback_written(&self, _written: &WrittenSession) -> anyhow::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CanonicalSession;
    use serde_json::json;

    fn session_with_id(id: &str) -> CanonicalSession {
        CanonicalSession {
            session_id: id.to_string(),
            provider_slug: "test".to_string(),
            workspace: None,
            title: None,
            started_at: None,
            ended_at: None,
            messages: vec![],
            metadata: json!({}),
            source_path: std::path::PathBuf::from("/tmp/x"),
            model_name: None,
        }
    }

    #[test]
    fn write_session_id_prefers_target_override() {
        let opts = WriteOptions {
            force: false,
            target_session_id: Some("deterministic-id".to_string()),
        };
        let id = resolve_write_session_id(&opts, &session_with_id("source-id"));
        assert_eq!(id, "deterministic-id");
    }

    #[test]
    fn write_session_id_sanitizes_untrusted_source_id() {
        let opts = WriteOptions {
            force: false,
            target_session_id: None,
        };
        for bad in ["../escape", "a/../../b", "..", "/abs/path", "a\\b"] {
            let id = resolve_write_session_id(&opts, &session_with_id(bad));
            assert!(
                !id.contains('/') && !id.contains('\\'),
                "unsanitized separator for {bad:?} -> {id:?}"
            );
            assert_ne!(id, "..");
            assert!(!id.is_empty());
        }
    }

    #[test]
    fn write_session_id_target_override_is_sanitized_too() {
        let opts = WriteOptions {
            force: false,
            target_session_id: Some("../pwned".to_string()),
        };
        let id = resolve_write_session_id(&opts, &session_with_id("anything"));
        assert_eq!(id, "..-pwned");
        assert!(!id.contains('/'));
    }
}

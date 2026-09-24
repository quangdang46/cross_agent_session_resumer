//! Cross-machine session transfer.
//!
//! A Claude Code session is not one file. Measured on a real machine
//! (session 5e492710, 2026-09-24):
//!
//! ```text
//! <session-uuid>.jsonl        27 MB   the transcript
//! <session-uuid>/tool-results 326 KB  tool output, referenced by the transcript
//! <session-uuid>/subagents    7.7 MB  child agent transcripts
//! ```
//!
//! `claude --resume` needs all three. It has no global lookup either — it
//! resolves the session inside `~/.claude/projects/<slug(cwd)>/`, so a transfer
//! to a different checkout path has to land in a different slug directory.
//!
//! This module produces a self-describing bundle that `unpack` can restore
//! anywhere. The transcript is truncated at the last *complete* JSON line:
//! the agent appends concurrently, and a byte-offset cut yields a torn tail
//! that fails to parse (casr already logs these as "skipping malformed JSON
//! line" while listing Codex sessions).

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::error::CasrError;

/// Bump when the on-disk layout changes in a way an older `unpack` cannot read.
pub const BUNDLE_VERSION: u32 = 1;

/// Claude Code encodes a cwd into a directory name by replacing every `/` and
/// `.` with `-`. Verified against the real store: `~/Projects/beads_rust`
/// became `-Users-tranquangdang21-Projects-beads_rust`.
///
/// The encoding is lossy — a literal `-` in a folder name is indistinguishable
/// from a separator — so the directory name is only ever a lookup key. The
/// authoritative path travels in `project.cwd` and is what the destination
/// machine substitutes its own root into.
pub fn slug_for_cwd(cwd: &Path) -> String {
    let mut out = String::with_capacity(cwd.as_os_str().len());
    for ch in cwd.to_string_lossy().chars() {
        match ch {
            '/' | '.' => out.push('-'),
            other => out.push(other),
        }
    }
    out
}

/// The projects directory Claude Code reads sessions from.
pub fn projects_root() -> Result<PathBuf, CasrError> {
    let Some(home) = dirs::home_dir() else {
        return Err(CasrError::SessionReadError {
            path: PathBuf::from("~/.claude/projects"),
            provider: "claude-code".into(),
            detail: "cannot determine home directory; set HOME".into(),
        });
    };
    let claude_dir = home.join(".claude").join("projects");
    if claude_dir.is_dir() {
        Ok(claude_dir)
    } else {
        Err(CasrError::ProviderUnavailable {
            provider: "claude-code".into(),
            reason: format!("no Claude Code session store at {}", claude_dir.display()),
            evidence: vec![format!("looked for {}", claude_dir.display())],
        })
    }
}

/// Copy `session.jsonl` while dropping a torn tail.
///
/// Keeps every line that parses as a standalone JSON value, so a bundle taken
/// while the agent is mid-write still restores. Returns the byte offset of the
/// last accepted line — the next incremental `pack` resumes from there instead
/// of re-sending the whole transcript.
pub fn copy_complete_lines(src: &Path, dest: &Path) -> Result<u64, CasrError> {
    let open = |p: &Path, action: &str| CasrError::SessionReadError {
        path: p.to_path_buf(),
        provider: "claude-code".into(),
        detail: format!("{action}: {}", p.display()),
    };
    let reader = BufReader::new(fs::File::open(src).map_err(|e| open(src, &e.to_string()))?);
    let mut out = fs::File::create(dest).map_err(|e| CasrError::SessionWriteError {
        path: dest.to_path_buf(),
        provider: "claude-code".into(),
        detail: e.to_string(),
    })?;
    let mut accepted: u64 = 0;

    for line in reader.split(b'\n') {
        let line = line.map_err(|e| open(src, &e.to_string()))?;
        // split() yields a final empty chunk for a trailing newline; skip it so
        // we never append a bare `\n` of our own.
        if line.is_empty() {
            continue;
        }
        if serde_json::from_slice::<serde_json::Value>(&line).is_err() {
            // Torn tail, or a non-JSON line we do not model. Stop here: bytes
            // after a bad line cannot be trusted to be in order either.
            break;
        }
        out.write_all(&line)
            .map_err(|e| CasrError::SessionWriteError {
                path: dest.to_path_buf(),
                provider: "claude-code".into(),
                detail: e.to_string(),
            })?;
        out.write_all(b"\n")
            .map_err(|e| CasrError::SessionWriteError {
                path: dest.to_path_buf(),
                provider: "claude-code".into(),
                detail: e.to_string(),
            })?;
        accepted += line.len() as u64 + 1;
    }

    out.flush().map_err(|e| CasrError::SessionWriteError {
        path: dest.to_path_buf(),
        provider: "claude-code".into(),
        detail: e.to_string(),
    })?;
    Ok(accepted)
}

/// Everything needed to restore one session on another machine.
#[derive(Debug, Serialize, Deserialize)]
pub struct Bundle {
    pub version: u32,
    pub provider: String,
    pub session_id: String,
    /// The cwd this session ran in, on the source machine.
    pub cwd: PathBuf,
    /// Session id → slice of a `tool-results/` or `subagents/` payload.
    pub sidecars: Vec<Sidecar>,
    /// Byte offset of the last complete line in the source transcript.
    pub transcript_offset: u64,
    pub transcript_bytes: u64,
    /// Uncommitted work, so the destination is not missing files the
    /// transcript claims to have edited.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git: Option<GitState>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Sidecar {
    /// Path relative to the session directory, e.g. `subagents/agent-x.jsonl`.
    pub rel: String,
    pub bytes: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GitState {
    pub branch: String,
    pub head: String,
    /// `git diff` output — portable, unlike the absolute paths in a diff stat.
    #[serde(default)]
    pub dirty_patch: String,
    pub untracked: Vec<String>,
}

impl Bundle {
    /// Where this session lives in the source machine's store.
    pub fn source_path(&self, root: &Path) -> PathBuf {
        root.join(slug_for_cwd(&self.cwd))
            .join(format!("{}.jsonl", self.session_id))
    }

    /// Where it must land on a machine whose checkout lives at `dest_cwd`.
    ///
    /// This is the whole of the cross-machine problem: same slug, copy; different
    /// slug, the destination computes its own.
    pub fn dest_path(&self, dest_root: &Path, dest_cwd: &Path) -> PathBuf {
        dest_root
            .join(slug_for_cwd(dest_cwd))
            .join(format!("{}.jsonl", self.session_id))
    }
}

/// Walk a session's sibling directory (`<uuid>/`) for payloads the transcript
/// references. Symlinks are skipped: a bundle that followed one could pull in
/// an entire home directory.
pub fn collect_sidecars(session_dir: &Path) -> Result<Vec<Sidecar>, CasrError> {
    let mut out = Vec::new();
    if !session_dir.is_dir() {
        return Ok(out);
    }
    for entry in WalkDir::new(session_dir).follow_links(false) {
        let entry = entry.map_err(|e| CasrError::SessionReadError {
            path: session_dir.to_path_buf(),
            provider: "claude-code".into(),
            detail: format!("walk: {e}"),
        })?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path
            .symlink_metadata()
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(false)
        {
            continue;
        }
        let rel = path
            .strip_prefix(session_dir)
            .map_err(|e| CasrError::SessionReadError {
                path: path.to_path_buf(),
                provider: "claude-code".into(),
                detail: format!("strip prefix: {e}"),
            })?
            .to_string_lossy()
            .to_string();
        let bytes = path
            .metadata()
            .map_err(|e| CasrError::SessionReadError {
                path: path.to_path_buf(),
                provider: "claude-code".into(),
                detail: e.to_string(),
            })?
            .len();
        out.push(Sidecar { rel, bytes });
    }
    out.sort_by(|a, b| a.rel.cmp(&b.rel));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_file(path: &Path, body: &str) {
        let mut f = fs::File::create(path).expect("create fixture");
        f.write_all(body.as_bytes()).expect("write fixture");
    }

    #[test]
    fn slug_replaces_separators_and_dots() {
        assert_eq!(
            slug_for_cwd(Path::new("/Users/me/Projects/beads_rust")),
            "-Users-me-Projects-beads_rust"
        );
        // A dotted directory loses the dot: verified against the real store,
        // where `~/code/foo.bar` is stored as `-Users-me-code-foo-bar`.
        assert_eq!(slug_for_cwd(Path::new("/a/b.c")), "-a-b-c");
    }

    #[test]
    fn slug_keeps_underscores() {
        // Distinct from the separator: `beads_rust` must not become `beads-rust`.
        assert_eq!(slug_for_cwd(Path::new("/p/beads_rust")), "-p-beads_rust");
    }

    #[test]
    fn dest_path_uses_destination_cwd_slug() {
        let b = Bundle {
            version: BUNDLE_VERSION,
            provider: "claude-code".into(),
            session_id: "abc".into(),
            cwd: PathBuf::from("/office/code/app"),
            sidecars: vec![],
            transcript_offset: 0,
            transcript_bytes: 0,
            git: None,
        };
        let root = Path::new("/home/me/.claude/projects");
        // Same path on both machines: the destination slug equals the source.
        assert_eq!(
            b.dest_path(root, Path::new("/office/code/app")),
            root.join("-office-code-app").join("abc.jsonl")
        );
        // Different checkout path: the destination computes its own slug. This
        // is the whole cross-machine problem, and it reduces to this line.
        assert_eq!(
            b.dest_path(root, Path::new("/home/me/app")),
            root.join("-home-me-app").join("abc.jsonl")
        );
    }

    #[test]
    fn copy_keeps_only_complete_json_lines() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = dir.path().join("s.jsonl");
        let dest = dir.path().join("d.jsonl");
        // Last line is a torn write: valid prefix, truncated JSON.
        write_file(&src, "{\"a\":1}\n{\"b\":2}\n{\"c\":3,\"d\":");
        let offset = copy_complete_lines(&src, &dest).expect("copy");
        let out = fs::read_to_string(&dest).expect("read back");
        assert_eq!(out, "{\"a\":1}\n{\"b\":2}\n");
        assert_eq!(offset, "{\"a\":1}\n{\"b\":2}\n".len() as u64);
    }

    #[test]
    fn copy_of_intact_file_preserves_every_line() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = dir.path().join("s.jsonl");
        let dest = dir.path().join("d.jsonl");
        write_file(&src, "{\"a\":1}\n{\"b\":2}\n");
        copy_complete_lines(&src, &dest).expect("copy");
        assert_eq!(
            fs::read_to_string(&dest).expect("read"),
            "{\"a\":1}\n{\"b\":2}\n"
        );
    }

    #[test]
    fn copy_of_empty_file_yields_empty_output() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = dir.path().join("s.jsonl");
        let dest = dir.path().join("d.jsonl");
        write_file(&src, "");
        assert_eq!(copy_complete_lines(&src, &dest).expect("copy"), 0);
        assert_eq!(fs::read_to_string(&dest).expect("read"), "");
    }

    #[test]
    fn copy_stops_at_first_bad_line_not_just_the_last() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = dir.path().join("s.jsonl");
        let dest = dir.path().join("d.jsonl");
        // A corrupt line in the middle: later bytes cannot be trusted to be in
        // order, so they must be dropped rather than silently kept.
        write_file(&src, "{\"a\":1}\nNOT JSON\n{\"b\":2}\n");
        copy_complete_lines(&src, &dest).expect("copy");
        assert_eq!(fs::read_to_string(&dest).expect("read"), "{\"a\":1}\n");
    }

    #[test]
    fn sidecars_are_collected_sorted_and_skip_symlinks() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sd = dir.path().join("sess");
        fs::create_dir_all(sd.join("subagents")).expect("mkdir");
        fs::create_dir_all(sd.join("tool-results")).expect("mkdir");
        write_file(&sd.join("tool-results/a.txt"), "a");
        write_file(&sd.join("subagents/b.jsonl"), "b");
        #[cfg(unix)]
        std::os::unix::fs::symlink("/etc/passwd", sd.join("evil")).expect("symlink");

        let got = collect_sidecars(&sd).expect("collect");
        let rels: Vec<&str> = got.iter().map(|s| s.rel.as_str()).collect();
        assert_eq!(rels, vec!["subagents/b.jsonl", "tool-results/a.txt"]);
    }

    #[test]
    fn sidecars_of_missing_dir_is_empty_not_an_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(
            collect_sidecars(&dir.path().join("nope"))
                .expect("collect")
                .is_empty()
        );
    }

    #[test]
    fn bundle_round_trips_through_json() {
        let b = Bundle {
            version: BUNDLE_VERSION,
            provider: "claude-code".into(),
            session_id: "s-1".into(),
            cwd: PathBuf::from("/w/app"),
            sidecars: vec![Sidecar {
                rel: "subagents/a.jsonl".into(),
                bytes: 12,
            }],
            transcript_offset: 4096,
            transcript_bytes: 27_547_554,
            git: Some(GitState {
                branch: "feat/x".into(),
                head: "deadbeef".into(),
                dirty_patch: "diff --git a b".into(),
                untracked: vec!["new.txt".into()],
            }),
        };
        let text = serde_json::to_string(&b).expect("serialize");
        let back: Bundle = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back.session_id, b.session_id);
        assert_eq!(back.transcript_offset, 4096);
        assert_eq!(back.git.expect("git").branch, "feat/x");
    }

    #[test]
    fn bundle_without_git_field_deserializes() {
        // Older bundles may omit the git block entirely; the destination must
        // still restore, just without the uncommitted-work claim.
        let text = r#"{"version":1,"provider":"claude-code","session_id":"s",
            "cwd":"/w","sidecars":[],"transcript_offset":0,"transcript_bytes":0}"#;
        let b: Bundle = serde_json::from_str(text).expect("deserialize");
        assert!(b.git.is_none());
    }
}

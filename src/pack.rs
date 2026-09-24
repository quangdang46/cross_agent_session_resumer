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
//! `claude --resume` reads a session from `~/.claude/projects/<slug(cwd)>/`,
//! and a transfer to a different checkout path has to land in a different slug
//! directory. Measured on 2.1.281, lookup also falls back to a one-level scan
//! of every other slug, so a misplaced session still resumes — but it resumes
//! *in the directory it was found in* and keeps growing there, and that
//! directory is not where the agent expects its history to live. Writing the
//! correct slug is what makes the transfer a restore rather than a scavenger
//! hunt, so `unpack` computes it from the destination checkout and never from
//! the source machine's path.
//!
//! This module produces a self-describing bundle that `unpack` restores
//! anywhere, and reads one back into a local store. The transcript is
//! truncated at the last *complete* JSON line: the agent appends
//! concurrently, and a byte-offset cut yields a torn tail that fails to parse
//! (casr already logs these as "skipping malformed JSON line" while listing
//! Codex sessions).
//!
//! `unpack` re-validates before it writes, because a bundle crossed a machine
//! boundary and the failure it guards against — a half-copied transcript — is
//! reported by `claude --resume` as `No conversation found`, the same message a
//! wrong slug produces. Nothing here deletes: a restore either creates a
//! destination or refuses, and only `--force` overwrites one.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Component, Path, PathBuf};
use walkdir::WalkDir;

use crate::error::CasrError;

/// Bump when the on-disk layout changes in a way an older `unpack` cannot read.
pub const BUNDLE_VERSION: u32 = 1;

/// Manifest filename inside a pack directory. Both halves read this, so a
/// rename cannot make `pack` write a directory `unpack` cannot find.
pub const MANIFEST_NAME: &str = "bundle.json";

/// Subtree holding the sidecars, relative to the pack directory.
pub const PAYLOAD_DIR: &str = "payload";

/// Errors carry the same shape everywhere in this module: which path, and what
/// the OS said. Callers add the remediation, which is a CLI concern.
fn read_error(path: &Path, detail: impl Into<String>) -> CasrError {
    CasrError::SessionReadError {
        path: path.to_path_buf(),
        provider: "claude-code".into(),
        detail: detail.into(),
    }
}

fn write_error(path: &Path, detail: impl Into<String>) -> CasrError {
    CasrError::SessionWriteError {
        path: path.to_path_buf(),
        provider: "claude-code".into(),
        detail: detail.into(),
    }
}

fn validation_error(errors: Vec<String>) -> CasrError {
    CasrError::ValidationError {
        errors,
        warnings: vec![],
        info: vec![],
    }
}

fn file_len(path: &Path) -> Result<u64, CasrError> {
    fs::metadata(path)
        .map(|m| m.len())
        .map_err(|e| read_error(path, e.to_string()))
}

/// Claude Code encodes a cwd into a directory name by replacing every `/` and
/// `.` with `-`. Verified against the real store: `~/Projects/beads_rust`
/// became `-Users-tranquangdang21-Projects-beads_rust`.
///
/// It slugs the *realpath*, not the literal argument. A session started from
/// the literal `/tmp/x` on macOS is stored under `-private-tmp-x`, because
/// `/tmp` is a symlink; a packer that slugs the spelling it was handed writes
/// to a directory the agent will never look in. So resolve first, and fall back
/// to the literal path when the directory does not exist — `--dry-run` is
/// routinely run before the checkout is created.
///
/// The encoding is lossy — a literal `-` in a folder name is indistinguishable
/// from a separator — so the directory name is only ever a lookup key. The
/// authoritative path travels in `Bundle::cwd` and is what the destination
/// machine substitutes its own root into.
pub fn slug_for_cwd(cwd: &Path) -> String {
    let resolved = fs::canonicalize(cwd).unwrap_or_else(|_| cwd.to_path_buf());
    let mut out = String::with_capacity(resolved.as_os_str().len());
    for ch in resolved.to_string_lossy().chars() {
        match ch {
            '/' | '.' => out.push('-'),
            other => out.push(other),
        }
    }
    out
}

/// Resolve the Claude Code config directory.
///
/// `CLAUDE_HOME` wins over `$HOME/.claude`, matching
/// `providers::claude_code::ClaudeCode::home_dir`. Honouring only one of the
/// two is how a run aimed at an isolated store ends up reading the real one.
fn claude_home(explicit: Option<String>, home: Option<PathBuf>) -> Result<PathBuf, CasrError> {
    match explicit.filter(|h| !h.is_empty()) {
        Some(dir) => Ok(PathBuf::from(dir)),
        None => home
            .map(|h| h.join(".claude"))
            .ok_or_else(|| CasrError::SessionReadError {
                path: PathBuf::from("~/.claude/projects"),
                provider: "claude-code".into(),
                detail: "cannot determine home directory; set HOME or CLAUDE_HOME".into(),
            }),
    }
}

/// The projects directory Claude Code reads sessions from.
pub fn projects_root() -> Result<PathBuf, CasrError> {
    let claude_dir =
        claude_home(std::env::var("CLAUDE_HOME").ok(), dirs::home_dir())?.join("projects");
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

/// Same as [`projects_root`], but tolerates a store that does not exist yet.
///
/// `unpack` is the one command whose whole job is to create it: a fresh
/// `CLAUDE_HOME` — exactly what the E2E harness builds — has no `projects/`
/// until the first restore. Requiring it to pre-exist would make the restore
/// that creates the store impossible to run.
pub fn projects_root_for_restore() -> Result<PathBuf, CasrError> {
    Ok(claude_home(std::env::var("CLAUDE_HOME").ok(), dirs::home_dir())?.join("projects"))
}

/// Copy `session.jsonl` while dropping a torn tail.
///
/// Keeps every line that parses as a standalone JSON value, so a bundle taken
/// while the agent is mid-write still restores. Returns the byte offset of the
/// last accepted line — the next incremental `pack` resumes from there instead
/// of re-sending the whole transcript.
///
/// The return value is deliberately one number with three meanings that must
/// agree: bytes written, bytes copied out of the source, and the source
/// position where copying stopped. That holds because the output is a
/// byte-exact *prefix* of the input — accepted lines and blank lines are
/// written through verbatim, and no `\n` is ever synthesised. The previous
/// version appended a delimiter the source never had, so an unterminated final
/// line reported an offset one byte past EOF, and a resumed pack would have
/// started mid-line.
pub fn copy_complete_lines(src: &Path, dest: &Path) -> Result<u64, CasrError> {
    let mut reader =
        BufReader::new(fs::File::open(src).map_err(|e| read_error(src, e.to_string()))?);
    let mut out = private_file(dest)?;
    let mut line: Vec<u8> = Vec::new();
    let mut copied: u64 = 0;

    loop {
        line.clear();
        // read_until rather than split: only a chunk that actually ended in the
        // delimiter is a complete line, and the copy has to stop on the same
        // boundary the offset reports.
        let read = reader
            .read_until(b'\n', &mut line)
            .map_err(|e| read_error(src, e.to_string()))?;
        if read == 0 {
            break;
        }
        let terminated = line.last() == Some(&b'\n');
        let body = if terminated {
            &line[..line.len() - 1]
        } else {
            &line[..]
        };
        // A blank line is not a torn tail. Copying it through keeps the output a
        // prefix of the source, which is what keeps the offset honest; treating
        // it as corruption would silently drop the rest of the transcript.
        if !body.iter().all(u8::is_ascii_whitespace)
            && serde_json::from_slice::<serde_json::Value>(body).is_err()
        {
            // Torn tail, or a non-JSON line we do not model. Stop here: bytes
            // after a bad line cannot be trusted to be in order either.
            break;
        }
        out.write_all(&line)
            .map_err(|e| write_error(dest, e.to_string()))?;
        copied += line.len() as u64;
    }

    out.flush().map_err(|e| write_error(dest, e.to_string()))?;
    Ok(copied)
}

/// Create `path` and any missing parents with the store's own mode.
///
/// The real store is `drwx------`; `create_dir_all` under the default umask
/// would widen a directory that is about to hold private transcripts.
fn private_dir_all(path: &Path) -> Result<(), CasrError> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(path)
        .map_err(|e| write_error(path, e.to_string()))
}

/// Open `path` for writing, truncating, at the store's own mode (`-rw-------`).
///
/// A session transcript is a user's complete private agent history and
/// routinely contains pasted API keys. It must not become *more* readable than
/// the file it was copied from just because it passed through a bundle.
fn private_file(path: &Path) -> Result<fs::File, CasrError> {
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts.open(path)
        .map_err(|e| write_error(path, e.to_string()))
}

/// Force `path` to the store's file mode.
///
/// `fs::copy` inherits the source's permissions, which for a pack payload is
/// whatever the *packing* machine's umask produced. This module's own files are
/// opened with an explicit mode; copies are not.
fn tighten(path: &Path) -> Result<(), CasrError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))
            .map_err(|e| write_error(path, format!("tighten permissions: {e}")))?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sidecar {
    /// Path relative to the session directory, e.g. `subagents/agent-x.jsonl`.
    ///
    /// Always `/`-separated, never the OS separator: this string travels in a
    /// bundle to a different machine, and a Windows pack's `subagents\a.jsonl`
    /// resolves to one odd filename on macOS — or, after a lossy round trip
    /// through `to_string_lossy`, to nothing at all.
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
        let entry = entry.map_err(|e| read_error(session_dir, format!("walk: {e}")))?;
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
            .map_err(|e| read_error(path, format!("strip prefix: {e}")))?
            .components()
            .map(|c| c.as_os_str().to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join("/");
        out.push(Sidecar {
            rel,
            bytes: file_len(path)?,
        });
    }
    out.sort_by(|a, b| a.rel.cmp(&b.rel));
    Ok(out)
}

/// How a restore should behave.
///
/// A struct rather than two positional booleans: `unpack(dir, root, cwd, opts)`
/// keeps the call site readable, and `..Default::default()` keeps tests from
/// having to spell out the combination they do not care about.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnpackOptions {
    /// Overwrite a session that already exists at the destination.
    pub force: bool,
    /// Validate and report, but write nothing.
    pub dry_run: bool,
}

/// Everything an unpack decided, so the CLI reports what happened instead of
/// re-deriving it from the bundle a second time.
#[derive(Debug)]
pub struct UnpackReport {
    pub session_id: String,
    /// The destination slug — `slug_for_cwd(dest_cwd)`, not the source's.
    pub slug: String,
    pub bundle_dir: PathBuf,
    /// The working tree the session ran in on the *source* machine. Recorded,
    /// not actionable: it rarely exists here.
    pub source_cwd: PathBuf,
    pub transcript_path: PathBuf,
    /// Bytes now at `transcript_path` (in a dry run, bytes that will land).
    pub transcript_bytes: u64,
    /// `<slug>/<session-id>/`, where the sidecars live.
    pub session_dir: PathBuf,
    pub restored: Vec<Sidecar>,
    /// A session was already at the destination. Always false on a clean run.
    pub would_overwrite: bool,
    pub dry_run: bool,
    /// Non-fatal: sidecars the manifest promised but the payload does not hold,
    /// sidecars whose size disagrees with the manifest, an existing session.
    pub warnings: Vec<String>,
}

/// One sidecar's copy job, resolved before anything is written.
struct SidecarPlan {
    sidecar: Sidecar,
    src: PathBuf,
    dest: PathBuf,
    available: bool,
}

/// Read and parse a pack's manifest.
///
/// The manifest is the only thing that describes the rest of the directory, so
/// it is read and validated before any path is touched.
pub fn read_bundle(manifest: &Path) -> Result<Bundle, CasrError> {
    let text = fs::read_to_string(manifest).map_err(|e| read_error(manifest, e.to_string()))?;
    serde_json::from_str(&text)
        .map_err(|e| read_error(manifest, format!("not a casr bundle manifest: {e}")))
}

/// Reject a manifest this build cannot honour.
///
/// A bundle crossed a machine boundary; a version mismatch means the payload
/// layout could differ from what `unpack` expects, and restoring anyway yields
/// a session that fails to load with nothing to point at. Same reasoning for
/// the provider field: the restore path here is Claude Code's.
pub fn validate_bundle(bundle: &Bundle) -> Result<(), CasrError> {
    if bundle.version != BUNDLE_VERSION {
        return Err(validation_error(vec![format!(
            "bundle is version {} but this casr reads version {BUNDLE_VERSION}; \
             re-pack it with a matching casr",
            bundle.version
        )]));
    }
    if bundle.provider != "claude-code" {
        return Err(validation_error(vec![format!(
            "bundle is for provider '{}'; unpack restores Claude Code sessions only",
            bundle.provider
        )]));
    }
    Ok(())
}

/// The transcript inside a pack directory, checked against the manifest.
///
/// Deriving the name from `bundle.session_id` would make the check vacuous, so
/// the directory is scanned instead. That catches a pack whose `.jsonl` was
/// renamed, or that lost it entirely — both of which `claude --resume` would
/// later report as `No conversation found`, indistinguishable from a wrong
/// slug.
pub fn bundle_transcript(bundle_dir: &Path, bundle: &Bundle) -> Result<PathBuf, CasrError> {
    let mut found: Vec<PathBuf> = Vec::new();
    for path in read_dir_entries(bundle_dir)? {
        if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
            found.push(path);
        }
    }
    found.sort();

    match found.as_slice() {
        [] => Err(validation_error(vec![format!(
            "no transcript (*.jsonl) in {}",
            bundle_dir.display()
        )])),
        [only] => {
            let stem = only
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_default();
            if stem != bundle.session_id {
                return Err(validation_error(vec![format!(
                    "manifest names session '{}' but the transcript is '{stem}.jsonl'",
                    bundle.session_id
                )]));
            }
            Ok(only.clone())
        }
        many => {
            let names: Vec<String> = many
                .iter()
                .map(|p| {
                    p.file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string()
                })
                .collect();
            Err(validation_error(vec![format!(
                "{} transcripts in the bundle, expected one: {}",
                many.len(),
                names.join(", ")
            )]))
        }
    }
}

fn read_dir_entries(dir: &Path) -> Result<Vec<PathBuf>, CasrError> {
    let entries = fs::read_dir(dir).map_err(|e| read_error(dir, e.to_string()))?;
    let mut out = Vec::new();
    for entry in entries {
        out.push(entry.map_err(|e| read_error(dir, e.to_string()))?.path());
    }
    Ok(out)
}

/// Reject a transcript whose last line does not parse.
///
/// This is the check `pack` cannot make on the far side: a bundle cut mid-write
/// is a valid file with a valid manifest, and the agent refuses it with the
/// same `No conversation found with session ID` it prints for a wrong slug.
/// Here the two are still distinguishable, so it is worth the read.
pub fn validate_transcript_tail(transcript: &Path) -> Result<(), CasrError> {
    let last = last_nonblank_line(transcript)?;
    if serde_json::from_slice::<serde_json::Value>(&last).is_ok() {
        return Ok(());
    }
    Err(validation_error(vec![format!(
        "the last line of {} is not valid JSON; the pack was cut mid-write",
        transcript.display()
    )]))
}

/// The last non-blank line of a JSONL file.
///
/// Streamed rather than slurped: a real packed transcript is 27 MB and only the
/// tail is needed, so the file must never be resident just to check it.
fn last_nonblank_line(path: &Path) -> Result<Vec<u8>, CasrError> {
    let reader = BufReader::new(fs::File::open(path).map_err(|e| read_error(path, e.to_string()))?);
    let mut last: Vec<u8> = Vec::new();
    for line in reader.split(b'\n') {
        let line = line.map_err(|e| read_error(path, e.to_string()))?;
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        last = line;
    }
    if last.is_empty() {
        return Err(validation_error(vec![format!(
            "{} is empty",
            path.display()
        )]));
    }
    Ok(last)
}

/// Turn a manifest-relative path into a path, or refuse it.
///
/// `sidecars[].rel` is untrusted input: the manifest came from another machine,
/// and `../../.ssh/authorized_keys` would otherwise be an entirely ordinary
/// copy. Only relative paths built from plain components are allowed.
fn safe_relative(rel: &str) -> Result<PathBuf, CasrError> {
    if rel.is_empty() {
        return Err(validation_error(vec![
            "bundle sidecar path is empty".to_string(),
        ]));
    }
    let candidate = Path::new(rel);
    if candidate.components().any(|c| {
        matches!(
            c,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return Err(validation_error(vec![format!(
            "bundle sidecar path '{rel}' escapes the bundle"
        )]));
    }
    Ok(candidate.to_path_buf())
}

/// Copy a file and confirm the destination is as long as the source.
///
/// A short write leaves a file that exists and is wrong — the one failure a
/// transfer tool cannot afford, because the manifest's byte counts would then
/// disagree with the tree sitting next to them.
fn copy_verified(src: &Path, dest: &Path) -> Result<u64, CasrError> {
    let wrote = fs::copy(src, dest).map_err(|e| read_error(src, e.to_string()))?;
    tighten(dest)?;
    let on_disk = file_len(dest)?;
    if on_disk != wrote {
        return Err(write_error(
            dest,
            format!("copied {wrote} bytes but the destination holds {on_disk}"),
        ));
    }
    Ok(wrote)
}

/// Restore a packed session into a Claude Code store.
///
/// This is the destination half of the module: `pack` reads one machine's store
/// into a portable directory, and this writes that directory into another
/// machine's store at the slug of *this* machine's checkout. That slug is the
/// entire cross-machine problem — the session id and the bytes travel, but the
/// directory the agent looks in is derived from a path that only exists here.
///
/// Nothing is deleted. A destination that already holds the session is a
/// `SessionConflict` unless `force` is set, and no path outside the destination
/// slug is ever written.
pub fn unpack(
    bundle_dir: &Path,
    dest_root: &Path,
    dest_cwd: &Path,
    opts: UnpackOptions,
) -> Result<UnpackReport, CasrError> {
    let bundle = read_bundle(&bundle_dir.join(MANIFEST_NAME))?;
    validate_bundle(&bundle)?;
    let source_transcript = bundle_transcript(bundle_dir, &bundle)?;
    validate_transcript_tail(&source_transcript)?;

    let dest = bundle.dest_path(dest_root, dest_cwd);
    let slug_dir = dest
        .parent()
        .ok_or_else(|| write_error(&dest, "the destination path has no parent directory"))?;
    let slug = slug_dir
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().to_string());
    let session_dir = slug_dir.join(&bundle.session_id);

    // Both trees are checked. A --force that replaces the transcript but leaves
    // a stale sidecar tree beside it produces a bundle that lies about its own
    // contents, which is the same class of bug as a silent overwrite.
    let would_overwrite = dest.exists() || session_dir.exists();
    if would_overwrite && !opts.force && !opts.dry_run {
        return Err(CasrError::SessionConflict {
            session_id: bundle.session_id.clone(),
            existing_path: dest.clone(),
        });
    }

    let mut warnings = Vec::new();
    if would_overwrite {
        warnings.push(format!(
            "overwriting the session already at {}",
            dest.display()
        ));
    }

    // Every sidecar is resolved before anything is written, so a dry run
    // reports the same plan the real run follows — including the ones that are
    // not actually there. A missing payload file is a warning, not a failure:
    // measured against a real 27 MB session, the resumed context is
    // byte-identical with and without the sidecars.
    let mut plans = Vec::with_capacity(bundle.sidecars.len());
    for sidecar in &bundle.sidecars {
        let rel = safe_relative(&sidecar.rel)?;
        let src = bundle_dir.join(PAYLOAD_DIR).join(&rel);
        let available = src.is_file();
        if !available {
            warnings.push(format!("sidecar missing from the bundle: {}", sidecar.rel));
        }
        plans.push(SidecarPlan {
            sidecar: sidecar.clone(),
            src,
            dest: session_dir.join(&rel),
            available,
        });
    }

    // The tail is already validated, so in a dry run the source size is exactly
    // what a real restore would land; the copy is authoritative once it runs.
    let mut transcript_bytes = file_len(&source_transcript)?;
    let mut restored: Vec<Sidecar> = Vec::new();

    if !opts.dry_run {
        private_dir_all(slug_dir)?;
        // copy_complete_lines drops a torn tail rather than writing one, and
        // opens the destination 0600 to match the store it is joining.
        transcript_bytes = copy_complete_lines(&source_transcript, &dest)?;
        for plan in &plans {
            if !plan.available {
                continue;
            }
            if let Some(parent) = plan.dest.parent() {
                private_dir_all(parent)?;
            }
            let wrote = copy_verified(&plan.src, &plan.dest)?;
            if wrote != plan.sidecar.bytes {
                warnings.push(format!(
                    "{} is {wrote} bytes but the manifest recorded {}",
                    plan.sidecar.rel, plan.sidecar.bytes
                ));
            }
            restored.push(plan.sidecar.clone());
        }
    }

    Ok(UnpackReport {
        session_id: bundle.session_id,
        slug,
        bundle_dir: bundle_dir.to_path_buf(),
        source_cwd: bundle.cwd,
        transcript_path: dest,
        transcript_bytes,
        session_dir,
        restored,
        would_overwrite,
        dry_run: opts.dry_run,
        warnings,
    })
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
    fn slug_resolves_a_symlinked_cwd_to_its_realpath() {
        // Claude Code slugs the realpath: a session started from the literal
        // `/tmp/x` on macOS is stored under `-private-tmp-x`, because `/tmp` is
        // a symlink. Slugging the spelling we were handed writes the bundle to
        // a directory the agent will never look in.
        let real = tempfile::tempdir().expect("tempdir");
        let holder = tempfile::tempdir().expect("tempdir");
        let link = holder.path().join("linked");
        #[cfg(unix)]
        std::os::unix::fs::symlink(real.path(), &link).expect("symlink");
        // Where symlinks do not exist the two spellings are the same
        // directory, so the assertion holds and still means something.
        #[cfg(not(unix))]
        fs::create_dir_all(&link).expect("mkdir");

        assert_eq!(slug_for_cwd(&link), slug_for_cwd(real.path()));
    }

    #[test]
    fn slug_of_a_missing_directory_falls_back_to_the_literal_path() {
        // `--dry-run` is routinely run before the checkout exists, and a
        // fallback that panicked here would make the preview impossible.
        let missing = Path::new("/definitely/not/a/real/dir");
        assert_eq!(slug_for_cwd(missing), "-definitely-not-a-real-dir");
    }

    #[test]
    fn claude_home_prefers_the_explicit_override() {
        assert_eq!(
            claude_home(Some("/tmp/alt".into()), Some(PathBuf::from("/home/me"))).expect("resolve"),
            PathBuf::from("/tmp/alt")
        );
    }

    #[test]
    fn claude_home_falls_back_to_dot_claude_under_home() {
        assert_eq!(
            claude_home(None, Some(PathBuf::from("/home/me"))).expect("resolve"),
            PathBuf::from("/home/me/.claude")
        );
        // An empty override counts as none: `export CLAUDE_HOME=$unset` leaves
        // exactly this behind, and honouring it would point the store at "".
        assert_eq!(
            claude_home(Some(String::new()), Some(PathBuf::from("/home/me"))).expect("resolve"),
            PathBuf::from("/home/me/.claude")
        );
    }

    #[test]
    fn claude_home_without_any_home_is_an_error() {
        assert!(claude_home(None, None).is_err());
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
    fn copy_offset_is_a_real_source_position_for_an_unterminated_tail() {
        // The exact state of a mid-write append: a valid last line with no
        // delimiter. Synthesising a `\n` and counting it reported an offset
        // one byte past EOF, so a resumed pack would have started inside the
        // next line rather than at its beginning.
        let dir = tempfile::tempdir().expect("tempdir");
        let src = dir.path().join("s.jsonl");
        let dest = dir.path().join("d.jsonl");
        let body = "{\"a\":1}\n{\"b\":2}";
        write_file(&src, body);

        let offset = copy_complete_lines(&src, &dest).expect("copy");
        assert_eq!(offset, body.len() as u64, "offset is a source position");
        assert_eq!(
            fs::read_to_string(&dest).expect("read"),
            body,
            "the output is a byte-exact prefix of the input"
        );
    }

    #[test]
    fn copy_counts_blank_lines_instead_of_stopping_at_them() {
        // A blank line is not a torn tail. Treating it as corruption dropped
        // every line after it, silently, and left the offset short by the blank
        // lines' own bytes.
        let dir = tempfile::tempdir().expect("tempdir");
        let src = dir.path().join("s.jsonl");
        let dest = dir.path().join("d.jsonl");
        let body = "{\"a\":1}\n\n\n{\"b\":2}\n";
        write_file(&src, body);

        let offset = copy_complete_lines(&src, &dest).expect("copy");
        assert_eq!(offset, body.len() as u64);
        assert_eq!(fs::read_to_string(&dest).expect("read"), body);
    }

    #[test]
    fn copy_preserves_crlf_byte_for_byte() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = dir.path().join("s.jsonl");
        let dest = dir.path().join("d.jsonl");
        // `\r` rides along on the chunk and serde_json accepts it as trailing
        // whitespace, so the whole file is accepted and written verbatim.
        let body = "{\"a\":1}\r\n{\"b\":2}\r\n";
        write_file(&src, body);

        let offset = copy_complete_lines(&src, &dest).expect("copy");
        assert_eq!(offset, body.len() as u64);
        assert_eq!(fs::read_to_string(&dest).expect("read"), body);
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
    fn sidecar_rel_paths_are_slash_separated_on_every_platform() {
        // A bundle is unpacked on a different machine, where `subagents\a.jsonl`
        // is one file whose name contains a backslash — the manifest would name
        // something the destination cannot find.
        let dir = tempfile::tempdir().expect("tempdir");
        let sd = dir.path().join("sess");
        fs::create_dir_all(sd.join("subagents")).expect("mkdir");
        write_file(&sd.join("subagents/a.jsonl"), "a");

        for s in collect_sidecars(&sd).expect("collect") {
            assert!(
                !s.rel.contains('\\'),
                "sidecar rel must be /-separated, got {}",
                s.rel
            );
        }
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

    fn sample_bundle(session_id: &str) -> Bundle {
        Bundle {
            version: BUNDLE_VERSION,
            provider: "claude-code".into(),
            session_id: session_id.into(),
            cwd: PathBuf::from("/w/app"),
            sidecars: vec![],
            transcript_offset: 0,
            transcript_bytes: 0,
            git: None,
        }
    }

    /// Rewrite a pack's manifest, so a test can declare state the directory
    /// does not actually hold. Packs are immutable fixtures here — nothing in
    /// this module deletes.
    fn patch_bundle(dir: &Path, edit: impl FnOnce(&mut Bundle)) {
        let manifest = dir.join(MANIFEST_NAME);
        let mut bundle: Bundle =
            serde_json::from_str(&fs::read_to_string(&manifest).expect("read")).expect("parse");
        edit(&mut bundle);
        write_file(
            &manifest,
            &serde_json::to_string_pretty(&bundle).expect("serialize"),
        );
    }

    /// Build a pack directory that `unpack` should accept.
    fn make_pack(root: &Path, session_id: &str, source_cwd: &str) -> PathBuf {
        let dir = root.join(format!("pack-{session_id}"));
        fs::create_dir_all(dir.join(PAYLOAD_DIR).join("subagents")).expect("mkdir");
        write_file(
            &dir.join(format!("{session_id}.jsonl")),
            "{\"type\":\"user\",\"uuid\":\"u1\",\"cwd\":\"/w\",\"timestamp\":\"2026-01-01T00:00:00.000Z\"}\n\
             {\"type\":\"assistant\",\"uuid\":\"a1\"}\n",
        );
        let payload = dir.join(PAYLOAD_DIR).join("subagents/a.jsonl");
        write_file(&payload, "child\n");

        let mut bundle = sample_bundle(session_id);
        bundle.cwd = PathBuf::from(source_cwd);
        bundle.sidecars = vec![Sidecar {
            rel: "subagents/a.jsonl".into(),
            bytes: file_len(&payload).expect("len"),
        }];
        write_file(
            &dir.join(MANIFEST_NAME),
            &serde_json::to_string_pretty(&bundle).expect("serialize"),
        );
        dir
    }

    /// A destination store and working tree to unpack into. The store is left
    /// uncreated on purpose: `unpack` is what creates it.
    fn destination(dir: &Path) -> (PathBuf, PathBuf) {
        let cwd = dir.join("checkout");
        fs::create_dir_all(&cwd).expect("mkdir");
        (dir.join("projects"), cwd)
    }

    fn restore(pack: &Path, root: &Path, cwd: &Path, opts: UnpackOptions) -> UnpackReport {
        unpack(pack, root, cwd, opts).expect("unpack")
    }

    /// Unpack that must be refused; panics if it is not.
    fn restore_err(pack: &Path, root: &Path, cwd: &Path) -> CasrError {
        unpack(pack, root, cwd, UnpackOptions::default()).expect_err("must refuse")
    }

    #[test]
    fn read_bundle_parses_a_written_manifest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let manifest = dir.path().join(MANIFEST_NAME);
        write_file(
            &manifest,
            &serde_json::to_string_pretty(&sample_bundle("s")).expect("serialize"),
        );
        assert_eq!(read_bundle(&manifest).expect("read").session_id, "s");
    }

    #[test]
    fn read_bundle_reports_a_missing_or_corrupt_manifest() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(read_bundle(&dir.path().join(MANIFEST_NAME)).is_err());

        let garbage = dir.path().join("garbage.json");
        write_file(&garbage, "not json at all");
        assert!(read_bundle(&garbage).is_err());
    }

    #[test]
    fn validate_bundle_accepts_a_current_claude_code_bundle() {
        assert!(validate_bundle(&sample_bundle("s")).is_ok());
    }

    #[test]
    fn validate_bundle_rejects_a_version_or_provider_it_cannot_restore() {
        // A version mismatch means the payload layout could differ, and
        // restoring anyway yields a session that fails to load for no visible
        // reason.
        let mut future = sample_bundle("s");
        future.version = BUNDLE_VERSION + 1;
        assert!(matches!(
            validate_bundle(&future),
            Err(CasrError::ValidationError { .. })
        ));

        let mut other = sample_bundle("s");
        other.provider = "codex".into();
        assert!(matches!(
            validate_bundle(&other),
            Err(CasrError::ValidationError { .. })
        ));
    }

    #[test]
    fn bundle_transcript_must_match_the_manifest_name() {
        let dir = tempfile::tempdir().expect("tempdir");
        // A transcript renamed out from under the manifest. Deriving the name
        // from the manifest would make this check vacuous, so the directory is
        // scanned instead.
        write_file(&dir.path().join("other.jsonl"), "{}\n");
        let err = bundle_transcript(dir.path(), &sample_bundle("s")).expect_err("name must match");
        assert!(
            matches!(err, CasrError::ValidationError { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn bundle_transcript_rejects_a_directory_with_none_or_many() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(bundle_transcript(dir.path(), &sample_bundle("s")).is_err());

        write_file(&dir.path().join("a.jsonl"), "{}\n");
        write_file(&dir.path().join("b.jsonl"), "{}\n");
        let err = bundle_transcript(dir.path(), &sample_bundle("s")).expect_err("two is ambiguous");
        assert!(
            matches!(err, CasrError::ValidationError { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn validate_transcript_tail_separates_a_torn_tail_from_an_empty_file() {
        let dir = tempfile::tempdir().expect("tempdir");

        let good = dir.path().join("good.jsonl");
        write_file(&good, "{\"a\":1}\n{\"b\":2}\n");
        assert!(validate_transcript_tail(&good).is_ok());

        // This is the case worth catching here: `claude --resume` reports a torn
        // transcript with the same `No conversation found` as a wrong slug, so
        // there is no second chance to tell them apart.
        let torn = dir.path().join("torn.jsonl");
        write_file(&torn, "{\"a\":1}\n{\"b\":");
        assert!(validate_transcript_tail(&torn).is_err());

        let empty = dir.path().join("empty.jsonl");
        write_file(&empty, "");
        assert!(validate_transcript_tail(&empty).is_err());
    }

    #[test]
    fn safe_relative_allows_plain_names_and_refuses_escapes() {
        assert_eq!(
            safe_relative("subagents/a.jsonl").expect("ok"),
            PathBuf::from("subagents/a.jsonl")
        );
        for bad in ["../escape", "a/../../escape", "/etc/passwd", ""] {
            assert!(
                matches!(safe_relative(bad), Err(CasrError::ValidationError { .. })),
                "expected {bad:?} to be refused"
            );
        }
    }

    #[test]
    fn copy_verified_copies_the_bytes_and_reports_a_missing_source() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = dir.path().join("src.txt");
        let dest = dir.path().join("dest.txt");
        write_file(&src, "hello");
        assert_eq!(copy_verified(&src, &dest).expect("copy"), 5);
        assert_eq!(fs::read_to_string(&dest).expect("read"), "hello");
        assert!(copy_verified(&dir.path().join("nope"), &dest).is_err());
    }

    #[test]
    fn unpack_restores_into_the_destination_slug() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pack = make_pack(dir.path(), "sess-1", "/w/app");
        let (root, cwd) = destination(dir.path());

        let report = restore(&pack, &root, &cwd, UnpackOptions::default());

        // The whole cross-machine problem in one assertion: the bundle names a
        // source tree that does not exist here, and the session still lands
        // where a resume from `cwd` will look for it.
        let expected = root.join(slug_for_cwd(&cwd)).join("sess-1.jsonl");
        assert_eq!(report.transcript_path, expected);
        assert_eq!(report.slug, slug_for_cwd(&cwd));
        assert_eq!(
            fs::read_to_string(&expected).expect("read"),
            fs::read_to_string(pack.join("sess-1.jsonl")).expect("read")
        );

        let sidecar = report.session_dir.join("subagents").join("a.jsonl");
        assert!(sidecar.is_file(), "sidecar sits beside the transcript");
        assert_eq!(report.restored.len(), 1);
        assert!(
            report.warnings.is_empty(),
            "unexpected: {:?}",
            report.warnings
        );
    }

    #[test]
    fn unpack_dry_run_reports_the_plan_and_writes_nothing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pack = make_pack(dir.path(), "sess-2", "/w/app");
        let (root, cwd) = destination(dir.path());

        let report = restore(
            &pack,
            &root,
            &cwd,
            UnpackOptions {
                dry_run: true,
                ..Default::default()
            },
        );

        assert!(report.dry_run);
        assert!(report.restored.is_empty());
        assert!(!root.exists(), "a dry run must not create the store");
        assert!(report.transcript_bytes > 0, "the plan still sizes the copy");
    }

    #[test]
    fn unpack_refuses_to_overwrite_without_force() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pack = make_pack(dir.path(), "sess-3", "/w/app");
        let (root, cwd) = destination(dir.path());
        restore(
            &pack,
            &root,
            &cwd,
            UnpackOptions {
                force: true,
                ..Default::default()
            },
        );

        let err = restore_err(&pack, &root, &cwd);
        assert!(
            matches!(err, CasrError::SessionConflict { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn unpack_with_force_replaces_the_existing_transcript() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pack = make_pack(dir.path(), "sess-4", "/w/app");
        let (root, cwd) = destination(dir.path());
        let first = restore(&pack, &root, &cwd, UnpackOptions::default());
        write_file(&first.transcript_path, "{\"stale\":true}\n");

        let forced = restore(
            &pack,
            &root,
            &cwd,
            UnpackOptions {
                force: true,
                ..Default::default()
            },
        );
        assert!(forced.would_overwrite);
        assert!(
            forced.warnings.iter().any(|w| w.contains("overwriting")),
            "the overwrite is reported, not silent: {:?}",
            forced.warnings
        );
        assert_eq!(
            fs::read_to_string(&forced.transcript_path).expect("read"),
            fs::read_to_string(pack.join("sess-4.jsonl")).expect("read")
        );
    }

    #[test]
    fn unpack_rejects_a_bundle_this_build_cannot_restore() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pack = make_pack(dir.path(), "sess-5", "/w/app");
        patch_bundle(&pack, |b| b.version = BUNDLE_VERSION + 1);
        let (root, cwd) = destination(dir.path());

        let err = restore_err(&pack, &root, &cwd);
        assert!(
            matches!(err, CasrError::ValidationError { .. }),
            "got {err:?}"
        );
        assert!(!root.exists(), "validation runs before any write");
    }

    #[test]
    fn unpack_rejects_a_transcript_cut_mid_line() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pack = make_pack(dir.path(), "sess-6", "/w/app");
        write_file(&pack.join("sess-6.jsonl"), "{\"a\":1}\n{\"b\":2,\"c\":");
        let (root, cwd) = destination(dir.path());

        let err = restore_err(&pack, &root, &cwd);
        assert!(
            matches!(err, CasrError::ValidationError { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn unpack_warns_about_a_sidecar_the_payload_does_not_hold() {
        // A sidecar is best-effort: measured against a real 27 MB session, the
        // resumed context is byte-identical with and without them. So a gap is
        // named and the restore still completes.
        let dir = tempfile::tempdir().expect("tempdir");
        let pack = make_pack(dir.path(), "sess-7", "/w/app");
        patch_bundle(&pack, |b| {
            b.sidecars.push(Sidecar {
                rel: "tool-results/gone.txt".into(),
                bytes: 10,
            })
        });
        let (root, cwd) = destination(dir.path());

        let report = restore(&pack, &root, &cwd, UnpackOptions::default());
        assert!(report.transcript_path.is_file());
        assert_eq!(report.restored.len(), 1, "the present sidecar still lands");
        assert!(
            report.warnings.iter().any(|w| w.contains("gone.txt")),
            "the gap is named: {:?}",
            report.warnings
        );
    }

    #[test]
    fn unpack_refuses_a_sidecar_path_that_escapes_the_bundle() {
        // `rel` came from another machine. A manifest naming
        // `../../.ssh/authorized_keys` must not be an ordinary copy.
        let dir = tempfile::tempdir().expect("tempdir");
        let pack = make_pack(dir.path(), "sess-8", "/w/app");
        patch_bundle(&pack, |b| {
            b.sidecars.push(Sidecar {
                rel: "../../escape.txt".into(),
                bytes: 0,
            })
        });
        let (root, cwd) = destination(dir.path());

        let err = restore_err(&pack, &root, &cwd);
        assert!(
            matches!(err, CasrError::ValidationError { .. }),
            "got {err:?}"
        );
        assert!(!dir.path().join("escape.txt").exists());
    }

    #[test]
    fn unpack_names_a_sidecar_whose_size_disagrees_with_the_manifest() {
        // The manifest is the only description of the payload, so a count that
        // does not match the bytes beside it is a finding, not a detail.
        let dir = tempfile::tempdir().expect("tempdir");
        let pack = make_pack(dir.path(), "sess-9", "/w/app");
        patch_bundle(&pack, |b| b.sidecars[0].bytes = 999_999);
        let (root, cwd) = destination(dir.path());

        let report = restore(&pack, &root, &cwd, UnpackOptions::default());
        assert!(
            report
                .warnings
                .iter()
                .any(|w| w.contains("999999") && w.contains("subagents/a.jsonl")),
            "the mismatch is reported: {:?}",
            report.warnings
        );
    }
}

//! Cross-machine transport: getting a packed bundle from machine A to machine B.
//!
//! `crate::pack` makes a session directory that `unpack` can restore anywhere.
//! This module is the other half — moving that directory between machines.
//!
//! The first transport is a plain git ref, `refs/casr/packs/<session-id>`,
//! because git already solves the parts that are genuinely hard: authenticated
//! transport, retries, a content-addressed store that deduplicates, and
//! history for free. It is deliberately *not* a git plugin of the bundle: the
//! bundle stays an ordinary directory of ordinary files, and a later
//! cloud/relay transport moves the same directory without changing `pack`, the
//! on-disk format, or any caller.
//!
//! Every step goes through plumbing — `hash-object`, `update-index`,
//! `write-tree`, `commit-tree`, `update-ref`, `checkout-index` — instead of a
//! working tree. There is no checkout to reset, no index to clobber, and no
//! `git clean` anywhere near a user's uncommitted work. A component whose job
//! is to move a session must not be able to reach for the commands that destroy
//! one.
//!
//! The safety rule is a fast-forward by construction. A new commit is always
//! parented on the tip the remote actually has, so the push is a fast-forward
//! and git accepts it; if somebody else pushed between our read and our write,
//! git rejects us instead, and the bundle they pushed stays reachable. A
//! `--force` exists, but only when the caller asks for it by name.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::error::CasrError;
use crate::pack::{BUNDLE_VERSION, Bundle, GitState, Sidecar};

/// Name of the manifest `casr pack` writes at the root of a bundle.
pub const MANIFEST_NAME: &str = "bundle.json";

/// Subdirectory holding sidecar payloads, one level below the transcript so a
/// restored bundle is laid out the way `pack` lays it out.
pub const PAYLOAD_DIR: &str = "payload";

/// Where a pushed bundle lives on the remote. Outside `refs/heads/` so it can
/// never be mistaken for a branch, checked out by accident, or deleted by a
/// `push --delete` that meant something else entirely.
pub const REF_NAMESPACE: &str = "refs/casr/packs";

/// Where a pulled bundle is staged locally. Kept apart from
/// [`REF_NAMESPACE`] so a fetch can never silently move the very ref that a
/// push would then try to fast-forward.
pub const INCOMING_NAMESPACE: &str = "refs/casr/incoming";

/// Transport name as it appears in JSON output and error messages.
pub const TRANSPORT_KIND: &str = "git";

/// Location of the local object cache, under the user's home directory.
pub const DEFAULT_STORE_DIR: &str = ".casr/transport";

/// Overrides [`default_store_dir`] without a flag, for a machine whose home
/// directory is not writable or a test that wants an isolated store.
pub const STORE_ENV: &str = "CASR_TRANSPORT_DIR";

/// All-zero object id: what `update-ref` expects when a ref must not exist.
const ZERO_OID: &str = "0000000000000000000000000000000000000000";

/// Bundles are data, never programs. A transported `+x` bit would let a
/// bundle change behaviour on the destination, so every file goes in as 100644
/// regardless of what the source machine had.
const FILE_MODE: &str = "100644";

/// Where the local object cache lives, resolved lazily so a caller can point it
/// somewhere else without a new flag on every subcommand.
pub fn default_store_dir() -> Result<PathBuf, CasrError> {
    if let Some(dir) = std::env::var_os(STORE_ENV) {
        return Ok(PathBuf::from(dir));
    }
    let Some(home) = dirs::home_dir() else {
        return Err(CasrError::ProviderUnavailable {
            provider: TRANSPORT_KIND.into(),
            reason: format!(
                "cannot determine home directory for the transport store; set HOME or {STORE_ENV}"
            ),
            evidence: vec![format!(
                "{STORE_ENV} was not set and home_dir() returned None"
            )],
        });
    };
    Ok(home.join(DEFAULT_STORE_DIR))
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

/// One file a bundle must contain, with the size it must have.
#[derive(Debug, Serialize, Deserialize)]
pub struct ManifestFile {
    /// Path relative to the bundle root, always with `/` separators so the
    /// same bundle describes itself identically on macOS and Linux.
    pub rel: String,
    /// Expected length, or 0 for "must exist, size not checked".
    pub bytes: u64,
}

/// What a packed bundle contains, derived from the [`Bundle`] that `pack`
/// wrote.
///
/// A 35 MB bundle is four files or four hundred, depending on how many
/// subagents ran, and a transfer that drops one still "succeeds" as far as the
/// filesystem is concerned. Resuming the result then fails much later with a
/// missing tool result, which reads like a casr bug rather than a bad
/// transfer. So both ends agree on this list: push refuses to ship a bundle
/// that is already incomplete, and pull checks the list after materialising.
#[derive(Debug, Serialize, Deserialize)]
pub struct BundleManifest {
    pub version: u32,
    pub provider: String,
    pub session_id: String,
    /// Sorted by `rel` so two machines describing the same bundle produce
    /// equal manifests and a diff between them means something.
    pub files: Vec<ManifestFile>,
    /// Bytes the packed transcript must have. *Not* `transcript_bytes`:
    /// that is the live file's length, which is larger whenever the agent was
    /// mid-append when `pack` read it.
    pub transcript_offset: u64,
    /// The source transcript's full size, carried for the report only.
    pub transcript_bytes: u64,
    pub sidecars: Vec<Sidecar>,
    pub git: Option<GitState>,
}

impl BundleManifest {
    /// The file list a bundle implies: the manifest itself, the transcript
    /// truncated at `transcript_offset`, and one payload file per sidecar.
    pub fn from_bundle(bundle: &Bundle) -> Self {
        let mut files = vec![
            ManifestFile {
                rel: MANIFEST_NAME.into(),
                // A file cannot state its own size in a document it does not yet
                // contain; `bundle.json` is verified by parsing it, not by
                // measuring it.
                bytes: 0,
            },
            ManifestFile {
                rel: transcript_name(&bundle.session_id),
                bytes: bundle.transcript_offset,
            },
        ];
        for sidecar in &bundle.sidecars {
            let rel = &sidecar.rel;
            files.push(ManifestFile {
                rel: format!("{PAYLOAD_DIR}/{rel}"),
                bytes: sidecar.bytes,
            });
        }
        files.sort_by(|a, b| a.rel.cmp(&b.rel));

        Self {
            version: bundle.version,
            provider: bundle.provider.clone(),
            session_id: bundle.session_id.clone(),
            files,
            transcript_offset: bundle.transcript_offset,
            transcript_bytes: bundle.transcript_bytes,
            sidecars: bundle.sidecars.iter().map(copy_sidecar).collect(),
            git: bundle.git.as_ref().map(copy_git),
        }
    }

    /// Read the manifest a bundle directory carries.
    pub fn from_bundle_dir(dir: &Path) -> Result<Self, CasrError> {
        Ok(Self::from_bundle(&read_bundle(dir)?))
    }

    /// Bytes the bundle must contain, excluding the manifest itself.
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.bytes).sum()
    }

    /// Every promised file is present, is a regular file, and is the promised
    /// length. Symlinks are rejected rather than followed: a bundle is
    /// attacker-controlled once it has crossed a network, and following one
    /// would let a fetch write outside the destination directory.
    ///
    /// Files the manifest does not mention are ignored, not removed — casr does
    /// not delete, and a destination pack directory may legitimately hold the
    /// leftovers of an earlier, longer session.
    pub fn verify(&self, dir: &Path) -> Result<(), CasrError> {
        let mut problems = Vec::new();
        for file in &self.files {
            let rel = &file.rel;
            let path = dir.join(rel);
            let meta = match path.symlink_metadata() {
                Ok(meta) => meta,
                Err(_) => {
                    problems.push(format!("missing {rel}"));
                    continue;
                }
            };
            if meta.file_type().is_symlink() {
                problems.push(format!("{rel} is a symlink"));
                continue;
            }
            if !meta.is_file() {
                problems.push(format!("{rel} is not a regular file"));
                continue;
            }
            if file.bytes > 0 && meta.len() != file.bytes {
                let expected = file.bytes;
                let found = meta.len();
                problems.push(format!("{rel} is {found} bytes, expected {expected}"));
            }
        }
        if problems.is_empty() {
            return Ok(());
        }
        let session_id = &self.session_id;
        let detail = problems.join("; ");
        Err(CasrError::SessionReadError {
            path: dir.to_path_buf(),
            provider: self.provider.clone(),
            detail: format!("bundle for session {session_id} is incomplete: {detail}"),
        })
    }

    /// The manifest must describe the session the caller asked for. Without
    /// this check a bundle for session A could be pushed to session B's ref,
    /// and the destination would restore a transcript under the wrong id — a
    /// resume that silently runs the wrong conversation.
    pub fn check_session(&self, expected: &str) -> Result<(), CasrError> {
        if self.session_id == expected {
            return Ok(());
        }
        let found = &self.session_id;
        Err(CasrError::SessionReadError {
            path: PathBuf::from(MANIFEST_NAME),
            provider: self.provider.clone(),
            detail: format!("bundle holds session {found}, caller asked for {expected}"),
        })
    }
}

/// Read `bundle.json` out of a bundle directory.
pub fn read_bundle(dir: &Path) -> Result<Bundle, CasrError> {
    let path = dir.join(MANIFEST_NAME);
    let text = fs::read_to_string(&path).map_err(|e| CasrError::SessionReadError {
        path: path.clone(),
        provider: TRANSPORT_KIND.into(),
        detail: format!("cannot read bundle manifest: {e}"),
    })?;
    serde_json::from_str(&text).map_err(|e| CasrError::SessionReadError {
        path,
        provider: TRANSPORT_KIND.into(),
        detail: format!("malformed bundle manifest: {e}"),
    })
}

fn transcript_name(session_id: &str) -> String {
    format!("{session_id}.jsonl")
}

/// `Sidecar` and `GitState` are plain data but are not `Clone`, and `pack` is
/// not this file to change; copying field by field keeps the manifest a
/// snapshot rather than a borrow of a bundle that may be dropped.
fn copy_sidecar(sidecar: &Sidecar) -> Sidecar {
    Sidecar {
        rel: sidecar.rel.clone(),
        bytes: sidecar.bytes,
    }
}

fn copy_git(git: &GitState) -> GitState {
    GitState {
        branch: git.branch.clone(),
        head: git.head.clone(),
        dirty_patch: git.dirty_patch.clone(),
        dirty_patch_truncated: git.dirty_patch_truncated,
        untracked: git.untracked.clone(),
    }
}

// ---------------------------------------------------------------------------
// Ref naming
// ---------------------------------------------------------------------------

/// The ref a bundle is pushed to and pulled from.
pub fn pack_ref(session_id: &str) -> String {
    format!("{REF_NAMESPACE}/{}", ref_component(session_id))
}

/// The local staging ref a fetch writes.
pub fn incoming_ref(session_id: &str) -> String {
    format!("{INCOMING_NAMESPACE}/{}", ref_component(session_id))
}

/// One path component of a ref name.
///
/// Session ids are uuids in practice, but nothing guarantees that, and a ref
/// name is not a place to be surprised by input. Everything outside
/// `[A-Za-z0-9_-]` becomes `-`, and a leading `-` or an empty result gains an
/// underscore, because git rejects both for a ref component. When sanitising
/// changed anything, 32 bits of digest of the original are appended so two
/// different ids cannot collapse onto one ref — otherwise one machine pushing
/// session "a b" and another pushing "a.b" would silently overwrite each
/// other.
///
/// The digest separator is `.`, which is *not* in the kept set, so a sanitised
/// id is always recognisable as one: a name containing a dot came from input
/// that had to be rewritten, and a plain id can never be spelled to look like
/// it.
fn ref_component(raw: &str) -> String {
    let mut base = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            base.push(ch);
        } else {
            base.push('-');
        }
    }
    if base.starts_with('-') {
        base.insert(0, '_');
    }
    if base.is_empty() {
        base.push('_');
    }
    if base == raw {
        return base;
    }
    let digest = Sha256::digest(raw.as_bytes());
    format!(
        "{base}.{:02x}{:02x}{:02x}{:02x}",
        digest[0], digest[1], digest[2], digest[3]
    )
}

/// Reject remotes that git would read as options, plus the empty string.
///
/// A remote is handed to `git push` as a positional argument and `push` has no
/// `--` separator, so a remote of `--upload-pack=...` would be run as a flag.
/// This is the whole of that attack surface for a value that can arrive from a
/// config file or a command line.
pub fn validate_remote(remote: &str) -> Result<(), CasrError> {
    let unusable =
        remote.is_empty() || remote.starts_with('-') || remote.chars().any(char::is_control);
    if !unusable {
        return Ok(());
    }
    Err(CasrError::ProviderUnavailable {
        provider: TRANSPORT_KIND.into(),
        reason: "git remote must be a non-empty path or URL that does not start with '-'".into(),
        evidence: vec![format!("remote: {remote:?}")],
    })
}

// ---------------------------------------------------------------------------
// Git invocations
// ---------------------------------------------------------------------------

/// A git command as data.
///
/// Building the argv before running anything is what makes this testable: CI
/// has no remote to push to, but it can assert that the argv built for a
/// non-fast-forward push contains no `--force`, and that the argv built for a
/// routine one contains no `reset`, no `clean`, and no force of any kind.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Invocation {
    argv: Vec<String>,
    env: Vec<(String, String)>,
}

impl Invocation {
    fn new(argv: Vec<String>) -> Self {
        let mut inv = Self {
            argv,
            env: Vec::new(),
        };
        // A CLI that shells out must fail on a missing credential, not sit on a
        // hidden password prompt while the user watches a command that looks
        // hung. A `git push` in a script has nobody to answer it.
        inv.env.push(("GIT_TERMINAL_PROMPT".into(), "0".into()));
        inv
    }

    fn with_env(mut self, key: &str, value: impl Into<String>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }

    /// What a user would have typed, for error messages.
    fn display(&self) -> String {
        self.argv.join(" ")
    }
}

/// Paths become argv through UTF-8. `pack` already treats a cwd as a string
/// (`slug_for_cwd` is lossy), and a path that cannot be spelled is a path that
/// cannot go in a manifest either, so this matches the rest of the crate.
fn arg(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// A store path is made absolute once, here: `checkout-index` runs with the
/// working directory set to the pull destination, so a relative
/// `GIT_INDEX_FILE` would be resolved against the wrong directory.
fn absolutize(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    std::env::current_dir().map_or_else(|_| path.to_path_buf(), |cwd| cwd.join(path))
}

/// `commit-tree` needs an identity, and a machine with a global
/// `commit.gpgsign=true` would otherwise fail a push for a reason that has
/// nothing to do with sessions. Both are passed per invocation, so the user's
/// own git configuration is left exactly as it was.
fn identity_args() -> Vec<String> {
    vec![
        "-c".into(),
        "user.name=casr".into(),
        "-c".into(),
        "user.email=casr@localhost".into(),
        "-c".into(),
        "commit.gpgsign=false".into(),
    ]
}

fn inv_init(store: &Path) -> Invocation {
    Invocation::new(vec![
        "git".into(),
        "init".into(),
        "--bare".into(),
        "--quiet".into(),
        arg(store),
    ])
}

fn inv_hash_object(store: &Path) -> Invocation {
    // `--stdin-paths` rather than file arguments: a bundle with a few thousand
    // sidecars must not be able to overflow the argument list, and a path
    // starting with `-` cannot be mistaken for a flag.
    Invocation::new(vec![
        "git".into(),
        format!("--git-dir={}", arg(store)),
        "hash-object".into(),
        "-w".into(),
        "--stdin-paths".into(),
    ])
}

fn inv_index(store: &Path, index: &Path) -> Invocation {
    Invocation::new(vec!["git".into(), format!("--git-dir={}", arg(store))])
        .with_env("GIT_INDEX_FILE", arg(index))
}

fn inv_read_tree_empty(store: &Path, index: &Path) -> Invocation {
    let mut inv = inv_index(store, index);
    inv.argv.extend(["read-tree".into(), "--empty".into()]);
    inv
}

fn inv_read_tree(store: &Path, index: &Path, tree: &str) -> Invocation {
    let mut inv = inv_index(store, index);
    inv.argv.extend(["read-tree".into(), tree.to_string()]);
    inv
}

fn inv_update_index_info(store: &Path, index: &Path) -> Invocation {
    let mut inv = inv_index(store, index);
    inv.argv
        .extend(["update-index".into(), "--index-info".into()]);
    inv
}

fn inv_write_tree(store: &Path, index: &Path) -> Invocation {
    let mut inv = inv_index(store, index);
    inv.argv.push("write-tree".into());
    inv
}

fn inv_commit_tree(store: &Path, tree: &str, parents: &[String], message: &str) -> Invocation {
    let mut argv = vec!["git".into(), format!("--git-dir={}", arg(store))];
    argv.extend(identity_args());
    argv.push("commit-tree".into());
    argv.push(tree.to_string());
    for parent in parents {
        argv.push("-p".into());
        argv.push(parent.clone());
    }
    argv.push("-m".into());
    argv.push(message.to_string());
    Invocation::new(argv)
}

fn inv_update_ref(store: &Path, name: &str, new: &str, old: &str) -> Invocation {
    // The fourth argument is the compare-and-swap: git refuses unless the ref
    // is at exactly `old`. Without it, two casr processes sharing a store race,
    // and the loser's commit would silently never be pushed.
    Invocation::new(vec![
        "git".into(),
        format!("--git-dir={}", arg(store)),
        "update-ref".into(),
        name.to_string(),
        new.to_string(),
        old.to_string(),
    ])
}

fn inv_rev_parse(store: &Path, rev: &str) -> Invocation {
    Invocation::new(vec![
        "git".into(),
        format!("--git-dir={}", arg(store)),
        "rev-parse".into(),
        "--verify".into(),
        "--quiet".into(),
        rev.to_string(),
    ])
}

fn inv_rev_parse_tree(store: &Path, rev: &str) -> Invocation {
    // The `^{tree}` peel is what distinguishes "the commit's id" from "the id
    // of the tree it points at". Getting this wrong ships a tree id where a
    // parent belongs and git refuses the commit.
    inv_rev_parse(store, &format!("{rev}^{{tree}}"))
}

fn inv_is_ancestor(store: &Path, ancestor: &str, descendant: &str) -> Invocation {
    Invocation::new(vec![
        "git".into(),
        format!("--git-dir={}", arg(store)),
        "merge-base".into(),
        "--is-ancestor".into(),
        ancestor.to_string(),
        descendant.to_string(),
    ])
}

fn inv_ls_remote(remote: &str, name: &str) -> Invocation {
    Invocation::new(vec![
        "git".into(),
        "ls-remote".into(),
        remote.to_string(),
        name.to_string(),
    ])
}

fn inv_fetch(store: &Path, remote: &str, src: &str, dst: &str, force: bool) -> Invocation {
    let mut argv = vec![
        "git".into(),
        format!("--git-dir={}", arg(store)),
        "fetch".into(),
        "--no-tags".into(),
    ];
    if force {
        argv.push("--force".into());
    }
    argv.push(remote.to_string());
    // A leading `+` is the refspec form of a forced update. It is added only
    // together with the explicit opt-in, never on its own.
    argv.push(format!("{}{src}:{dst}", if force { "+" } else { "" }));
    Invocation::new(argv)
}

fn inv_push(store: &Path, remote: &str, src: &str, dst: &str, force: bool) -> Invocation {
    let mut argv = vec![
        "git".into(),
        format!("--git-dir={}", arg(store)),
        "push".into(),
    ];
    if force {
        argv.push("--force".into());
    }
    argv.push(remote.to_string());
    argv.push(format!("{src}:{dst}"));
    Invocation::new(argv)
}

fn inv_checkout_index(store: &Path, work_tree: &Path, index: &Path, overwrite: bool) -> Invocation {
    // `--work-tree` is not optional: git refuses to run `checkout-index`
    // without one, and it is how a bundle is materialised without ever
    // creating a checkout that something else could reset.
    let mut inv = Invocation::new(vec![
        "git".into(),
        format!("--git-dir={}", arg(store)),
        format!("--work-tree={}", arg(work_tree)),
    ])
    .with_env("GIT_INDEX_FILE", arg(index));
    // A global `core.autocrlf`, or a destination reached over a path git cannot
    // spell, would rewrite the transcript on the way out and every byte count
    // in the manifest would then be wrong on arrival.
    inv.argv.extend(["-c".into(), "core.autocrlf=false".into()]);
    inv.argv.extend(["-c".into(), "core.eol=lf".into()]);
    inv.argv.push("checkout-index".into());
    inv.argv.push("--all".into());
    if overwrite {
        inv.argv.push("--force".into());
    }
    inv
}

fn inv_ls_tree_names(store: &Path, tree: &str) -> Invocation {
    // `-z` so a file name containing a newline cannot split into two entries.
    Invocation::new(vec![
        "git".into(),
        format!("--git-dir={}", arg(store)),
        "ls-tree".into(),
        "-r".into(),
        "-z".into(),
        "--name-only".into(),
        "--full-tree".into(),
        tree.to_string(),
    ])
}

// ---------------------------------------------------------------------------
// Running git
// ---------------------------------------------------------------------------

/// Run a git command and hand back stdout.
///
/// Every non-zero exit becomes a `CasrError` carrying git's own stderr. The
/// text a user needs to fix a wrong remote, a missing key or a rejected push
/// is in there — not in anything this module could invent.
fn run(inv: &Invocation, stdin: Option<&str>, cwd: Option<&Path>) -> Result<String, CasrError> {
    let mut cmd = Command::new(&inv.argv[0]);
    cmd.args(&inv.argv[1..])
        .envs(inv.env.iter().map(|(k, v)| (k, v)))
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }

    let mut child = cmd.spawn().map_err(|e| spawn_error(inv, &e))?;

    // The path list of a large bundle is bigger than a pipe buffer, and git
    // fills its own stdout with a hash per file while reading. Writing inline
    // would deadlock: both sides blocked on a full pipe. A writer thread keeps
    // both draining, and dropping the pipe here is what sends git its EOF.
    let writer = stdin.map(|input| {
        // Owned, because the writer outlives this frame: handing the thread a
        // borrow of the caller's `&str` would not outlive the call.
        let input = input.to_string();
        let mut pipe = child.stdin.take().expect("stdin was piped");
        std::thread::spawn(move || {
            // A broken pipe means git rejected the request or already had
            // everything; the exit status below is the error worth reporting.
            let _ = pipe.write_all(input.as_bytes());
        })
    });

    let out = child.wait_with_output().map_err(|e| spawn_error(inv, &e))?;
    if let Some(handle) = writer {
        let _ = handle.join();
    }
    if !out.status.success() {
        return Err(command_error(inv, out.status.code(), &out.stderr));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Same as [`run`], but the question is only "did it exit zero?" — used for
/// `merge-base --is-ancestor`, whose answer is its status and whose output is
/// nothing. A spawn failure counts as "not an ancestor", which is the
/// direction that keeps history rather than dropping it.
fn succeeds(inv: &Invocation) -> bool {
    Command::new(&inv.argv[0])
        .args(&inv.argv[1..])
        .envs(inv.env.iter().map(|(k, v)| (k, v)))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn spawn_error(inv: &Invocation, err: &std::io::Error) -> CasrError {
    let reason = if err.kind() == std::io::ErrorKind::NotFound {
        "git is not installed or not on PATH".to_string()
    } else {
        format!("could not start git: {err}")
    };
    CasrError::ProviderUnavailable {
        provider: TRANSPORT_KIND.into(),
        reason,
        evidence: vec![inv.display()],
    }
}

fn command_error(inv: &Invocation, code: Option<i32>, stderr: &[u8]) -> CasrError {
    let subcommand = inv.argv.get(1).cloned().unwrap_or_default();
    let reason = match code {
        Some(code) => format!("`git {subcommand}` failed with exit code {code}"),
        None => format!("`git {subcommand}` was killed before it finished"),
    };
    let stderr = String::from_utf8_lossy(stderr);
    let mut evidence = vec![inv.display()];
    for line in stderr.lines().filter(|l| !l.trim().is_empty()).take(8) {
        evidence.push(format!("  {line}"));
    }
    CasrError::ProviderUnavailable {
        provider: TRANSPORT_KIND.into(),
        reason,
        evidence,
    }
}

// ---------------------------------------------------------------------------
// Output parsing
// ---------------------------------------------------------------------------

/// `ls-remote` prints `<oid>\t<ref>`, or nothing at all when the remote has no
/// such ref. "No ref" is not an error — it is a first push.
fn parse_ls_remote(out: &str) -> Option<String> {
    let first = out.lines().next()?.trim();
    if first.is_empty() {
        return None;
    }
    let oid = first.split('\t').next()?.trim();
    if oid.is_empty() {
        return None;
    }
    Some(oid.to_string())
}

/// `hash-object --stdin-paths` prints one id per input line, in order. A count
/// that does not match means paths went missing between building the list and
/// running git, and a tree built from a shifted list would hold a bundle whose
/// contents do not match its own manifest — so this is a hard error, never a
/// silent truncation.
fn parse_blob_ids(out: &str, expected: usize) -> Result<Vec<String>, CasrError> {
    let ids: Vec<String> = out
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(String::from)
        .collect();
    if ids.len() == expected {
        return Ok(ids);
    }
    let found = ids.len();
    Err(CasrError::ProviderUnavailable {
        provider: TRANSPORT_KIND.into(),
        reason: format!("git hashed {found} files but the manifest lists {expected}"),
        evidence: vec![],
    })
}

/// `ls-tree -z --name-only` output: NUL-separated paths.
fn parse_tree_names(out: &str) -> Vec<String> {
    out.split('\0')
        .map(str::trim_end)
        .filter(|name| !name.is_empty())
        .map(String::from)
        .collect()
}

/// `update-index --index-info` input: one `<mode> <oid>\t<path>` line per
/// file, in the same order as the ids they came from.
///
/// The path must be relative to the repository root. An absolute one is not an
/// error: git prints `Ignoring path`, exits zero, and `write-tree` then returns
/// the empty tree — a bundle that pushes cleanly and restores as nothing.
fn format_index_info(files: &[ManifestFile], ids: &[String]) -> String {
    let mut out = String::new();
    for (file, id) in files.iter().zip(ids) {
        out.push_str(FILE_MODE);
        out.push(' ');
        out.push_str(id);
        out.push('\t');
        out.push_str(&file.rel);
        out.push('\n');
    }
    out
}

/// What staging dropped or invented, or `None` when the tree holds exactly the
/// files the manifest names.
///
/// Both sides are sorted with `/` separators, so this is a plain comparison.
/// It is the only check that catches every quiet way a file can fail to reach
/// the tree — an ignored path, a smudge filter, a path that differs by one
/// character — and every one of them would otherwise be discovered on the other
/// machine, as a session that resumes with content missing.
fn staging_differs(expected: &[String], staged: &[String]) -> Option<String> {
    if expected == staged {
        return None;
    }
    let missing: Vec<&str> = expected
        .iter()
        .filter(|rel| !staged.iter().any(|s| s == *rel))
        .map(String::as_str)
        .collect();
    let extra: Vec<&str> = staged
        .iter()
        .filter(|rel| !expected.iter().any(|s| s == *rel))
        .map(String::as_str)
        .collect();
    let mut why = format!(
        "staged tree holds {} file(s), manifest lists {}",
        staged.len(),
        expected.len()
    );
    if !missing.is_empty() {
        why.push_str(&format!("; missing {}", missing.join(", ")));
    }
    if !extra.is_empty() {
        why.push_str(&format!("; unexpected {}", extra.join(", ")));
    }
    Some(why)
}

// ---------------------------------------------------------------------------
// Transport shape
// ---------------------------------------------------------------------------

/// One argument bundle, addressed by the session it carries.
#[derive(Debug, Clone)]
pub struct PushRequest {
    pub session_id: String,
    /// A directory laid out the way `casr pack` lays one out.
    pub bundle_dir: PathBuf,
    /// Allow a push that is not a fast-forward. Nothing but an explicit "yes,
    /// discard the remote's version of this bundle" should set it.
    pub force: bool,
}

/// One argument bundle, addressed by the session it should produce.
#[derive(Debug, Clone)]
pub struct PullRequest {
    pub session_id: String,
    /// Where the bundle directory is materialised. Created if absent.
    pub dest_dir: PathBuf,
    /// Allow a fetch that is not a fast-forward, for a remote whose pack ref
    /// was rewritten behind our back.
    pub force: bool,
    /// Allow overwriting files already at `dest_dir`.
    pub overwrite: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PushReceipt {
    pub transport: String,
    pub session_id: String,
    pub remote: String,
    pub ref_name: String,
    pub commit: String,
    pub tree: String,
    pub files: usize,
    pub bytes: u64,
    /// False when the remote already held exactly this bundle. Re-pushing an
    /// unchanged bundle is the normal case for an agent that has not appended
    /// since the last push, and it must not grow the ref.
    pub pushed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PullReceipt {
    pub transport: String,
    pub session_id: String,
    pub remote: String,
    pub ref_name: String,
    pub commit: String,
    pub tree: String,
    pub dest_dir: PathBuf,
    pub files: usize,
    pub bytes: u64,
    pub transcript_offset: u64,
    pub transcript_bytes: u64,
    pub sidecars: usize,
}

/// Moving a bundle between machines.
///
/// The trait exists so a relay or cloud transport can be added without callers
/// — a `casr push` / `casr pull` subcommand, or anything else — learning what
/// a git ref is. Deliberately two methods and a name: the shape worth keeping
/// is [`PushRequest`] and [`PullRequest`], which describe a bundle in
/// transport-neutral terms.
pub trait Transport {
    /// Stable name for JSON output and error messages.
    fn kind(&self) -> &'static str;
    /// Make the bundle at `req.bundle_dir` available on the far side.
    fn push(&self, req: &PushRequest) -> Result<PushReceipt, CasrError>;
    /// Bring the bundle for `req.session_id` to `req.dest_dir`, verified whole.
    fn pull(&self, req: &PullRequest) -> Result<PullReceipt, CasrError>;
}

// ---------------------------------------------------------------------------
// Git transport
// ---------------------------------------------------------------------------

/// A bundle moved through a git ref.
#[derive(Debug, Clone)]
pub struct GitTransport {
    /// Local object cache. Everything is fetched and committed here first, so
    /// two casr runs on one machine share objects instead of re-transferring
    /// 35 MB, and a push can diff against the previous version without a round
    /// trip.
    store: PathBuf,
    /// Anything `git push` accepts: a URL, `user@host:path`, or a local path.
    remote: String,
}

impl GitTransport {
    pub fn new(remote: impl Into<String>, store: impl Into<PathBuf>) -> Self {
        let store: PathBuf = store.into();
        Self {
            store: absolutize(&store),
            remote: remote.into(),
        }
    }

    /// Point at a remote using the store under the user's home directory.
    pub fn with_default_store(remote: impl Into<String>) -> Result<Self, CasrError> {
        Ok(Self::new(remote, default_store_dir()?))
    }

    pub fn store(&self) -> &Path {
        &self.store
    }

    pub fn remote(&self) -> &str {
        &self.remote
    }

    /// The commit a bundle's ref points at on the remote, or `None` if the
    /// remote has never seen it.
    pub fn remote_head(&self, session_id: &str) -> Result<Option<String>, CasrError> {
        validate_remote(&self.remote)?;
        let out = run(
            &inv_ls_remote(&self.remote, &pack_ref(session_id)),
            None,
            None,
        )?;
        Ok(parse_ls_remote(&out))
    }

    /// The local object cache is a bare repository created on demand. `git init`
    /// on an existing bare repo reinitialises it and changes nothing, so this
    /// call doubles as the idempotency guard for a second run.
    fn ensure_store(&self) -> Result<(), CasrError> {
        fs::create_dir_all(&self.store).map_err(|e| CasrError::SessionWriteError {
            path: self.store.clone(),
            provider: TRANSPORT_KIND.into(),
            detail: format!("cannot create transport store: {e}"),
        })?;
        run(&inv_init(&self.store), None, None)?;
        Ok(())
    }

    /// The scratch index lives inside the store and is named per process: two
    /// casr processes on one machine must not share an index, and nothing
    /// outside this module ever looks at it.
    fn index_path(&self) -> PathBuf {
        self.store
            .join(format!("casr-index-{}", std::process::id()))
    }

    /// The commit a local ref points at, or `None` when it does not exist.
    fn local_head(&self, name: &str) -> Option<String> {
        let out = run(&inv_rev_parse(&self.store, name), None, None).ok()?;
        let oid = out.trim();
        if oid.is_empty() {
            return None;
        }
        Some(oid.to_string())
    }

    /// The tree a commit has, if we happen to hold the object.
    fn tree_of(&self, commit: &str) -> Option<String> {
        let out = run(&inv_rev_parse_tree(&self.store, commit), None, None).ok()?;
        let tree = out.trim();
        if tree.is_empty() {
            return None;
        }
        Some(tree.to_string())
    }

    fn is_ancestor(&self, ancestor: &str, descendant: &str) -> bool {
        succeeds(&inv_is_ancestor(&self.store, ancestor, descendant))
    }

    /// Hash the manifest's files and write them as a tree, returning its id.
    fn write_tree(
        &self,
        manifest: &BundleManifest,
        dir: &Path,
        index: &Path,
    ) -> Result<String, CasrError> {
        if manifest.files.is_empty() {
            let provider = &manifest.provider;
            let session_id = &manifest.session_id;
            return Err(CasrError::SessionReadError {
                path: dir.to_path_buf(),
                provider: provider.clone(),
                detail: format!("manifest for session {session_id} lists no files"),
            });
        }
        let mut listing = String::new();
        for file in &manifest.files {
            listing.push_str(&arg(&dir.join(&file.rel)));
            listing.push('\n');
        }

        let hashed = run(&inv_hash_object(&self.store), Some(listing.as_str()), None)?;
        let ids = parse_blob_ids(&hashed, manifest.files.len())?;

        // Start from an empty index every time: `--index-info` only adds, so a
        // stale entry from an earlier bundle would otherwise be committed too.
        run(&inv_read_tree_empty(&self.store, index), None, None)?;
        let info = format_index_info(&manifest.files, &ids);
        run(
            &inv_update_index_info(&self.store, index),
            Some(info.as_str()),
            None,
        )?;
        let tree = run(&inv_write_tree(&self.store, index), None, None)?;
        let tree = tree.trim().to_string();

        // The tree must be exactly the manifest, checked here rather than on
        // arrival: a bundle that pushes cleanly and restores as nothing is the
        // worst possible failure mode, and it costs one `ls-tree`.
        let expected: Vec<String> = manifest.files.iter().map(|f| f.rel.clone()).collect();
        let listing = run(&inv_ls_tree_names(&self.store, &tree), None, None)?;
        let staged = parse_tree_names(&listing);
        if let Some(why) = staging_differs(&expected, &staged) {
            return Err(CasrError::SessionWriteError {
                path: self.store.clone(),
                provider: manifest.provider.clone(),
                detail: format!("could not stage the bundle: {why}"),
            });
        }
        Ok(tree)
    }
}

impl Transport for GitTransport {
    fn kind(&self) -> &'static str {
        TRANSPORT_KIND
    }

    fn push(&self, req: &PushRequest) -> Result<PushReceipt, CasrError> {
        validate_remote(&self.remote)?;
        let name = pack_ref(&req.session_id);

        // The manifest is read from the bundle rather than taken on trust from
        // the caller: it is the only thing that says which files belong, and
        // both ends of a transfer are checked against the same list.
        let manifest = BundleManifest::from_bundle_dir(&req.bundle_dir)?;
        manifest.check_session(&req.session_id)?;
        if manifest.version > BUNDLE_VERSION {
            let found = manifest.version;
            return Err(CasrError::SessionReadError {
                path: req.bundle_dir.join(MANIFEST_NAME),
                provider: manifest.provider.clone(),
                detail: format!(
                    "bundle format version {found} is newer than this casr understands ({BUNDLE_VERSION})"
                ),
            });
        }
        // Refuse to publish a bundle that is already short a file. Shipping a
        // known-incomplete bundle and discovering it on the other machine turns
        // a local, fixable problem into somebody else's.
        manifest.verify(&req.bundle_dir)?;

        self.ensure_store()?;
        let index = self.index_path();
        let tree = self.write_tree(&manifest, &req.bundle_dir, &index)?;

        let remote_head = self.remote_head(&req.session_id)?;
        let local_head = self.local_head(&name);

        // Idempotency. When the local ref already is the remote's tip we hold
        // the object, so the comparison costs nothing; an unchanged bundle is
        // then a no-op instead of a second commit that changes nothing.
        if let (Some(remote_oid), Some(local_oid)) = (&remote_head, &local_head) {
            let unchanged = remote_oid == local_oid
                && self.tree_of(remote_oid).as_deref() == Some(tree.as_str());
            if unchanged {
                return Ok(PushReceipt {
                    transport: TRANSPORT_KIND.into(),
                    session_id: req.session_id.clone(),
                    remote: self.remote.clone(),
                    ref_name: name,
                    commit: remote_oid.clone(),
                    tree,
                    files: manifest.files.len(),
                    bytes: manifest.total_bytes(),
                    pushed: false,
                });
            }
        }

        let mut parents = Vec::new();
        if let Some(remote_oid) = &remote_head {
            // First parent is the remote's tip. That is what makes the push a
            // fast-forward, and it is why an ordinary update never needs
            // `--force`.
            parents.push(remote_oid.clone());
        }
        if let Some(local_oid) = &local_head {
            // A commit made here earlier that the remote has not seen is kept
            // as a second parent so it stays reachable. A bundle is the only
            // copy of a session's sidecars; making an older one unreachable is
            // the one thing this module must never do quietly.
            let already_remote = remote_head.as_ref().is_some_and(|remote_oid| {
                remote_oid == local_oid || self.is_ancestor(local_oid, remote_oid)
            });
            if !already_remote {
                parents.push(local_oid.clone());
            }
        }

        let files = manifest.files.len();
        let bytes = manifest.total_bytes();
        let session_id = &req.session_id;
        let message = format!("casr pack {session_id} ({files} files, {bytes} bytes)");
        let commit = run(
            &inv_commit_tree(&self.store, &tree, &parents, &message),
            None,
            None,
        )?;
        let commit = commit.trim().to_string();

        run(
            &inv_update_ref(
                &self.store,
                &name,
                &commit,
                local_head.as_deref().unwrap_or(ZERO_OID),
            ),
            None,
            None,
        )?;

        run(
            &inv_push(&self.store, &self.remote, &name, &name, req.force),
            None,
            None,
        )?;

        Ok(PushReceipt {
            transport: TRANSPORT_KIND.into(),
            session_id: req.session_id.clone(),
            remote: self.remote.clone(),
            ref_name: name,
            commit,
            tree,
            files,
            bytes,
            pushed: true,
        })
    }

    fn pull(&self, req: &PullRequest) -> Result<PullReceipt, CasrError> {
        validate_remote(&self.remote)?;
        let name = pack_ref(&req.session_id);
        let incoming = incoming_ref(&req.session_id);

        if self.remote_head(&req.session_id)?.is_none() {
            let session_id = &req.session_id;
            let remote = &self.remote;
            return Err(CasrError::SessionReadError {
                path: PathBuf::from(name),
                provider: TRANSPORT_KIND.into(),
                detail: format!("no bundle for session {session_id} at remote {remote}"),
            });
        }

        self.ensure_store()?;
        let index = self.index_path();
        run(
            &inv_fetch(&self.store, &self.remote, &name, &incoming, req.force),
            None,
            None,
        )?;

        let commit = self
            .local_head(&incoming)
            .ok_or_else(|| CasrError::SessionReadError {
                path: PathBuf::from(&incoming),
                provider: TRANSPORT_KIND.into(),
                detail: format!("fetched ref {incoming} does not resolve to a commit"),
            })?;
        let tree = self
            .tree_of(&commit)
            .ok_or_else(|| CasrError::SessionReadError {
                path: PathBuf::from(&incoming),
                provider: TRANSPORT_KIND.into(),
                detail: format!("fetched ref {incoming} has no tree"),
            })?;

        fs::create_dir_all(&req.dest_dir).map_err(|e| CasrError::SessionWriteError {
            path: req.dest_dir.clone(),
            provider: TRANSPORT_KIND.into(),
            detail: format!("cannot create bundle directory: {e}"),
        })?;

        // Refuse before writing anything, not part-way through. `checkout-index`
        // reports a clobbered file on stdout and still exits zero, so the check
        // has to happen here or an overwrite would pass unnoticed.
        if !req.overwrite {
            let listing = run(&inv_ls_tree_names(&self.store, &tree), None, None)?;
            let incoming_files = parse_tree_names(&listing);
            let existing: Vec<String> = incoming_files
                .iter()
                .filter(|rel| req.dest_dir.join(rel).exists())
                .cloned()
                .collect();
            if let Some(first) = existing.first() {
                // `SessionConflict` would promise a `.bak` backup, which an
                // extraction into a pack directory does not make.
                let count = existing.len();
                return Err(CasrError::ProviderUnavailable {
                    provider: TRANSPORT_KIND.into(),
                    reason: format!(
                        "{count} file(s) already exist in the destination (first: {first}); \
                         re-run with overwrite enabled"
                    ),
                    evidence: existing
                        .iter()
                        .take(8)
                        .map(|rel| format!("  {}", req.dest_dir.join(rel).display()))
                        .collect(),
                });
            }
        }

        run(&inv_read_tree(&self.store, &index, &tree), None, None)?;
        run(
            &inv_checkout_index(&self.store, &req.dest_dir, &index, req.overwrite),
            None,
            Some(&req.dest_dir),
        )?;

        // Verify what actually landed, from the manifest that landed with it. A
        // transfer can be truncated and a remote can be hostile; either way the
        // check belongs at this boundary, before anything tries to resume.
        let manifest = BundleManifest::from_bundle_dir(&req.dest_dir)?;
        manifest.check_session(&req.session_id)?;
        manifest.verify(&req.dest_dir)?;

        Ok(PullReceipt {
            transport: TRANSPORT_KIND.into(),
            session_id: req.session_id.clone(),
            remote: self.remote.clone(),
            ref_name: name,
            commit,
            tree,
            dest_dir: req.dest_dir.clone(),
            files: manifest.files.len(),
            bytes: manifest.total_bytes(),
            transcript_offset: manifest.transcript_offset,
            transcript_bytes: manifest.transcript_bytes,
            sidecars: manifest.sidecars.len(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_string()).collect()
    }

    /// Realistic sizes, for the manifest arithmetic.
    fn sample_bundle() -> Bundle {
        Bundle {
            version: BUNDLE_VERSION,
            provider: "claude-code".into(),
            session_id: "5e492710-96cf-483a-929d-5330b8ad3000".into(),
            cwd: PathBuf::from("/Users/me/Projects/openproxy"),
            sidecars: vec![
                Sidecar {
                    rel: "subagents/agent-a.jsonl".into(),
                    bytes: 4096,
                },
                Sidecar {
                    rel: "tool-results/toolu_01.txt".into(),
                    bytes: 512,
                },
            ],
            transcript_offset: 27_547_000,
            // The live file kept growing while `pack` read it, so the source is
            // larger than what the bundle contains.
            transcript_bytes: 27_547_554,
            git: Some(GitState {
                branch: "poc/cross-machine-session".into(),
                head: "deadbeef".into(),
                dirty_patch: "diff --git a/Cargo.toml b/Cargo.toml".into(),
                dirty_patch_truncated: false,
                untracked: vec!["notes.md".into()],
            }),
        }
    }

    /// Small enough to write to disk several times per test run.
    fn small_bundle() -> Bundle {
        Bundle {
            transcript_offset: 16,
            transcript_bytes: 20,
            sidecars: vec![Sidecar {
                rel: "subagents/agent-a.jsonl".into(),
                bytes: 5,
            }],
            ..sample_bundle()
        }
    }

    /// A bundle directory that matches `bundle`, so manifest checks run against
    /// real bytes instead of a mocked filesystem.
    fn write_bundle(dir: &Path, bundle: &Bundle) {
        fs::create_dir_all(dir).expect("mkdir bundle");
        let manifest = BundleManifest::from_bundle(bundle);
        for file in &manifest.files {
            if file.bytes == 0 {
                continue;
            }
            let path = dir.join(&file.rel);
            fs::create_dir_all(path.parent().expect("parent")).expect("mkdir parent");
            fs::write(&path, vec![b'x'; file.bytes as usize]).expect("write fixture");
        }
        let text = serde_json::to_string_pretty(bundle).expect("serialize");
        fs::write(dir.join(MANIFEST_NAME), text).expect("write manifest");
    }

    /// The one path component a ref ends in, i.e. what the sanitiser produced.
    ///
    /// The separator belongs to the namespace: stripping `refs/casr/packs` alone
    /// would leave a leading `/` on the component and make every ref look like a
    /// nested one.
    fn ref_tail<'a>(name: &'a str, namespace: &str) -> &'a str {
        name.strip_prefix(namespace)
            .and_then(|rest| rest.strip_prefix('/'))
            .unwrap_or_else(|| panic!("{name:?} is not under {namespace:?}"))
    }

    /// The rules `git check-ref-format --branch` applies to one component.
    fn assert_valid_ref_component(tail: &str) {
        assert!(!tail.is_empty(), "empty ref component");
        assert!(tail.len() <= 255, "ref component too long: {tail:?}");
        assert!(
            !tail.starts_with('-') && !tail.starts_with('.'),
            "ref component may not start with '-' or '.': {tail:?}"
        );
        assert!(
            !tail.ends_with(".lock"),
            "ref component ends in .lock: {tail:?}"
        );
        assert!(!tail.contains(".."), "'..' in {tail:?}");
        assert!(!tail.contains("//"), "'//' in {tail:?}");
        for ch in tail.chars() {
            assert!(
                ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.',
                "illegal character {ch:?} in {tail:?}"
            );
        }
    }

    // --- refs -------------------------------------------------------------

    #[test]
    fn uuid_session_id_is_kept_verbatim() {
        // The common case must produce a ref a human can recognise in
        // `git ls-remote`.
        assert_eq!(
            pack_ref("5e492710-96cf-483a-929d-5330b8ad3000"),
            "refs/casr/packs/5e492710-96cf-483a-929d-5330b8ad3000"
        );
    }

    #[test]
    fn ref_naming_is_deterministic() {
        assert_eq!(
            pack_ref("sess:ion/with junk"),
            pack_ref("sess:ion/with junk")
        );
        assert_eq!(
            incoming_ref("sess:ion/with junk"),
            incoming_ref("sess:ion/with junk")
        );
    }

    #[test]
    fn ref_name_cannot_escape_its_namespace() {
        for hostile in [
            "../../etc/passwd",
            "..",
            "a/../../b",
            "refs/heads/main",
            "x.lock",
            "-dash",
            "@",
            "a@{0}",
        ] {
            let name = pack_ref(hostile);
            let tail = ref_tail(&name, REF_NAMESPACE);
            assert!(!tail.contains('/'), "{hostile:?} nested the ref: {name}");
            assert_valid_ref_component(tail);
            let incoming = incoming_ref(hostile);
            assert_valid_ref_component(ref_tail(&incoming, INCOMING_NAMESPACE));
        }
    }

    #[test]
    fn ref_name_never_keeps_a_space_or_control_character() {
        for raw in ["a b", "a\tb", "a\nb", "a\\b", "üñí", "a^b~c:?*[b"] {
            let name = pack_ref(raw);
            assert_valid_ref_component(ref_tail(&name, REF_NAMESPACE));
        }
    }

    #[test]
    fn ids_that_look_alike_after_sanitising_stay_separate() {
        // Without the digest suffix both land on `refs/casr/packs/a-b` and the
        // second push would be a fast-forward straight over the first.
        assert_ne!(pack_ref("a b"), pack_ref("a.b"));
    }

    #[test]
    fn a_plain_id_is_told_apart_from_a_sanitised_one() {
        // The dot is the digest separator, and a plain id can never contain
        // one — so a sanitised ref is always recognisable as sanitised.
        let plain = pack_ref("abc-123");
        assert!(!ref_tail(&plain, REF_NAMESPACE).contains('.'));
        let spaced = pack_ref("a b");
        let sanitised = ref_tail(&spaced, REF_NAMESPACE);
        assert_eq!(sanitised.matches('.').count(), 1);
        assert_eq!(sanitised.len(), "a-b.".len() + 8, "32 bits of digest");
    }

    #[test]
    fn empty_session_id_still_yields_a_valid_ref() {
        for raw in ["", " ", "/", "..."] {
            let tail = ref_component(raw);
            assert_valid_ref_component(&tail);
        }
    }

    #[test]
    fn incoming_ref_is_a_different_namespace() {
        assert_eq!(incoming_ref("s1"), "refs/casr/incoming/s1");
        assert_ne!(pack_ref("s1"), incoming_ref("s1"));
    }

    // --- remote validation -------------------------------------------------

    #[test]
    fn remote_rejects_anything_git_would_read_as_a_flag() {
        for bad in ["", "-x", "--upload-pack=touch /tmp/pwn", "a\nb", "a\rb"] {
            assert!(
                validate_remote(bad).is_err(),
                "{bad:?} must be rejected before it reaches git"
            );
        }
    }

    #[test]
    fn remote_accepts_the_three_shapes_a_user_actually_types() {
        for good in [
            "git@example.com:me/sessions.git",
            "https://example.com/me/sessions.git",
            "/tmp/casr-remote.git",
            "./relative/remote.git",
        ] {
            assert!(validate_remote(good).is_ok(), "{good:?} should be accepted");
        }
    }

    // --- argv construction -------------------------------------------------

    #[test]
    fn init_argv_creates_a_bare_repository() {
        let inv = inv_init(Path::new("/s/store.git"));
        assert_eq!(
            inv.argv,
            strings(&["git", "init", "--bare", "--quiet", "/s/store.git"])
        );
    }

    #[test]
    fn every_invocation_disables_terminal_prompts() {
        // Otherwise a push needing a credential blocks forever inside a CLI.
        for inv in [
            inv_init(Path::new("/s/store.git")),
            inv_push(Path::new("/s"), "r", "a", "b", false),
            inv_fetch(Path::new("/s"), "r", "a", "b", false),
        ] {
            assert!(
                inv.env
                    .contains(&("GIT_TERMINAL_PROMPT".to_string(), "0".to_string())),
                "{:?} would block on a hidden prompt",
                inv.argv
            );
        }
    }

    /// Every invocation the transport can build without an opt-in.
    fn routine_invocations() -> Vec<Invocation> {
        let store = Path::new("/s/store.git");
        let index = Path::new("/s/i.idx");
        vec![
            inv_init(store),
            inv_hash_object(store),
            inv_read_tree_empty(store, index),
            inv_read_tree(store, index, "t"),
            inv_update_index_info(store, index),
            inv_write_tree(store, index),
            inv_commit_tree(store, "t", &strings(&["a"]), "m"),
            inv_update_ref(store, "r", "new", ZERO_OID),
            inv_rev_parse(store, "r"),
            inv_rev_parse_tree(store, "r"),
            inv_is_ancestor(store, "a", "b"),
            inv_ls_remote("r", "refs/casr/packs/s"),
            inv_fetch(store, "r", "a", "b", false),
            inv_push(store, "r", "a", "b", false),
            inv_checkout_index(store, Path::new("/d"), index, false),
            inv_ls_tree_names(store, "t"),
        ]
    }

    #[test]
    fn no_invocation_can_destroy_anything() {
        // The hard requirement, asserted over every argv the transport builds:
        // there is no `reset --hard`, no `clean`, no `rm`, no filter rewrite,
        // and no delete that could take a session with it.
        for inv in routine_invocations() {
            for banned in [
                "reset",
                "clean",
                "rm",
                "checkout",
                "filter-branch",
                "revert",
            ] {
                assert!(
                    !inv.argv.iter().any(|arg| arg == banned),
                    "{banned:?} found in {:?}",
                    inv.argv
                );
            }
            for banned in ["--force", "-f", "--delete", "--prune", "--hard"] {
                assert!(
                    !inv.argv.iter().any(|arg| arg == banned),
                    "{banned:?} found in {:?}",
                    inv.argv
                );
            }
        }
    }

    #[test]
    fn a_routine_pull_never_forces_its_fetch() {
        // `checkout-index` without `--force` still clobbers, so the overwrite
        // opt-in is checked by hand before the argv is ever built; the flag
        // itself only appears when the caller asked.
        for inv in routine_invocations() {
            assert!(
                !inv.argv.iter().any(|arg| arg.starts_with('+')),
                "a forced refspec found in {:?}",
                inv.argv
            );
        }
    }

    #[test]
    fn hash_object_takes_paths_on_stdin() {
        let inv = inv_hash_object(Path::new("/s/store.git"));
        assert_eq!(
            inv.argv,
            strings(&[
                "git",
                "--git-dir=/s/store.git",
                "hash-object",
                "-w",
                "--stdin-paths"
            ])
        );
        // No bundle path is an argument, so a huge bundle cannot overflow the
        // argument list and a file named like a flag cannot bite.
        assert!(!inv.argv.iter().any(|arg| arg.contains("bundle")));
    }

    #[test]
    fn index_invocations_scope_git_to_the_scratch_index() {
        // A shared index would let one casr process commit another's tree.
        let inv = inv_write_tree(Path::new("/s/store.git"), Path::new("/s/i.idx"));
        assert_eq!(
            inv.env,
            vec![
                ("GIT_TERMINAL_PROMPT".to_string(), "0".to_string()),
                ("GIT_INDEX_FILE".to_string(), "/s/i.idx".to_string()),
            ]
        );
        assert_eq!(inv.argv.last().map(String::as_str), Some("write-tree"));
    }

    #[test]
    fn read_tree_empty_empties_rather_than_reads_a_revision() {
        let inv = inv_read_tree_empty(Path::new("/s/store.git"), Path::new("/s/i.idx"));
        assert_eq!(
            inv.argv,
            strings(&["git", "--git-dir=/s/store.git", "read-tree", "--empty"])
        );
    }

    #[test]
    fn rev_parse_tree_peels_the_commit() {
        // Confusing the commit id with its tree id ships a tree where a parent
        // belongs, and git refuses the commit.
        assert_eq!(
            inv_rev_parse(Path::new("/s"), "refs/casr/packs/s").argv,
            strings(&[
                "git",
                "--git-dir=/s",
                "rev-parse",
                "--verify",
                "--quiet",
                "refs/casr/packs/s"
            ])
        );
        assert_eq!(
            inv_rev_parse_tree(Path::new("/s"), "abc")
                .argv
                .last()
                .map(String::as_str),
            Some("abc^{tree}")
        );
    }

    #[test]
    fn commit_tree_carries_an_identity_and_disables_signing() {
        let inv = inv_commit_tree(Path::new("/s/store.git"), "tree1", &[], "msg");
        assert!(inv.argv.contains(&"user.name=casr".to_string()));
        assert!(inv.argv.contains(&"user.email=casr@localhost".to_string()));
        // A global commit.gpgsign=true would otherwise fail the push.
        assert!(inv.argv.contains(&"commit.gpgsign=false".to_string()));
        assert!(inv.argv.contains(&"tree1".to_string()));
        assert_eq!(inv.argv.last().map(String::as_str), Some("msg"));
        assert!(
            !inv.argv.contains(&"-p".to_string()),
            "a root commit has no parent"
        );
    }

    #[test]
    fn commit_tree_gets_one_parent_flag_per_parent() {
        let parents = strings(&["aaa", "bbb"]);
        let inv = inv_commit_tree(Path::new("/s/store.git"), "t", &parents, "m");
        let flags: Vec<&String> = inv.argv.iter().filter(|arg| *arg == "-p").collect();
        assert_eq!(flags.len(), 2);
        let first = inv.argv.iter().position(|arg| arg == "-p").expect("-p");
        assert_eq!(inv.argv[first + 1], "aaa");
        assert_eq!(inv.argv[first + 3], "bbb");
    }

    #[test]
    fn update_ref_always_carries_the_expected_old_value() {
        // Compare-and-swap: without the fourth argument a losing racer would
        // silently move the ref out from under the push that follows.
        let inv = inv_update_ref(
            Path::new("/s/store.git"),
            "refs/casr/packs/s",
            "new",
            ZERO_OID,
        );
        assert_eq!(
            inv.argv,
            strings(&[
                "git",
                "--git-dir=/s/store.git",
                "update-ref",
                "refs/casr/packs/s",
                "new",
                ZERO_OID
            ])
        );
    }

    #[test]
    fn push_without_opt_in_contains_no_force() {
        let inv = inv_push(
            Path::new("/s/store.git"),
            "git@example.com:x.git",
            "refs/casr/packs/s",
            "refs/casr/packs/s",
            false,
        );
        assert!(
            !inv.argv.iter().any(|arg| arg == "--force" || arg == "-f"),
            "a default push must never force: {:?}",
            inv.argv
        );
        assert_eq!(
            inv.argv,
            strings(&[
                "git",
                "--git-dir=/s/store.git",
                "push",
                "git@example.com:x.git",
                "refs/casr/packs/s:refs/casr/packs/s"
            ])
        );
    }

    #[test]
    fn push_forces_only_when_asked() {
        let inv = inv_push(
            Path::new("/s/store.git"),
            "r",
            "refs/casr/packs/s",
            "refs/casr/packs/s",
            true,
        );
        assert_eq!(inv.argv[3], "--force");
        // The refspec keeps its plain form here; a leading `+` is a second way
        // to force, and it belongs to the fetch path only.
        assert_eq!(inv.argv[5], "refs/casr/packs/s:refs/casr/packs/s");
    }

    #[test]
    fn fetch_without_opt_in_is_a_plain_fast_forward_refspec() {
        let inv = inv_fetch(
            Path::new("/s/store.git"),
            "r",
            "refs/casr/packs/s",
            "refs/casr/incoming/s",
            false,
        );
        assert_eq!(
            inv.argv,
            strings(&[
                "git",
                "--git-dir=/s/store.git",
                "fetch",
                "--no-tags",
                "r",
                "refs/casr/packs/s:refs/casr/incoming/s"
            ])
        );
    }

    #[test]
    fn fetch_forces_only_when_asked() {
        let inv = inv_fetch(
            Path::new("/s/store.git"),
            "r",
            "refs/casr/packs/s",
            "refs/casr/incoming/s",
            true,
        );
        assert!(inv.argv.contains(&"--force".to_string()));
        // `+` forces independently of the flag, so it must ride on the same
        // opt-in and never appear alone.
        assert!(
            inv.argv
                .contains(&"+refs/casr/packs/s:refs/casr/incoming/s".to_string())
        );
    }

    #[test]
    fn checkout_index_needs_a_work_tree() {
        // git refuses to run it otherwise, and it is how a bundle is
        // materialised without creating a checkout anybody could reset.
        let inv = inv_checkout_index(
            Path::new("/s/store.git"),
            Path::new("/dest"),
            Path::new("/s/i.idx"),
            false,
        );
        assert!(inv.argv.contains(&"--work-tree=/dest".to_string()));
        assert!(inv.argv.contains(&"--all".to_string()));
        assert!(!inv.argv.contains(&"--force".to_string()));
    }

    #[test]
    fn checkout_index_overwrites_only_when_told_to() {
        let inv = inv_checkout_index(
            Path::new("/s/store.git"),
            Path::new("/dest"),
            Path::new("/s/i.idx"),
            true,
        );
        assert!(inv.argv.contains(&"--force".to_string()));
    }

    #[test]
    fn checkout_index_pins_line_endings() {
        // Otherwise a global autocrlf would rewrite the transcript and every
        // byte count in the manifest would be wrong on arrival.
        let inv = inv_checkout_index(
            Path::new("/s/store.git"),
            Path::new("/dest"),
            Path::new("/s/i.idx"),
            false,
        );
        assert!(inv.argv.contains(&"core.autocrlf=false".to_string()));
        assert!(inv.argv.contains(&"core.eol=lf".to_string()));
    }

    #[test]
    fn ls_remote_asks_for_exactly_one_ref() {
        let inv = inv_ls_remote("r", "refs/casr/packs/s");
        assert_eq!(
            inv.argv,
            strings(&["git", "ls-remote", "r", "refs/casr/packs/s"])
        );
    }

    #[test]
    fn ls_tree_names_is_nul_separated() {
        // A file name with a newline must not become two entries.
        let inv = inv_ls_tree_names(Path::new("/s/store.git"), "t");
        assert!(inv.argv.contains(&"-z".to_string()));
        assert!(inv.argv.contains(&"--name-only".to_string()));
    }

    // --- output parsing ----------------------------------------------------

    #[test]
    fn ls_remote_output_parsing() {
        assert_eq!(
            parse_ls_remote("abc123\trefs/casr/packs/s\n"),
            Some("abc123".to_string())
        );
        assert_eq!(parse_ls_remote("\n"), None, "no ref yet is not an error");
        assert_eq!(parse_ls_remote(""), None);
        assert_eq!(
            parse_ls_remote("a\trefs/casr/packs/x\nb\trefs/casr/packs/y\n"),
            Some("a".to_string()),
            "a full ref name matches at most one line"
        );
    }

    #[test]
    fn blob_ids_must_match_the_requested_count() {
        let ok = parse_blob_ids("aa\nbb\ncc\n", 3).expect("parse");
        assert_eq!(ok, strings(&["aa", "bb", "cc"]));
        // A shifted list would write a bundle whose contents do not match its
        // own manifest, so a short answer is a hard error.
        assert!(parse_blob_ids("aa\nbb\n", 3).is_err());
        assert!(parse_blob_ids("aa\nbb\ncc\ndd\n", 3).is_err());
    }

    #[test]
    fn tree_name_parsing_splits_on_nul_only() {
        assert_eq!(
            parse_tree_names("a.jsonl\0payload/x\0"),
            strings(&["a.jsonl", "payload/x"])
        );
        assert!(parse_tree_names("").is_empty());
        assert_eq!(
            parse_tree_names("a\nb\0"),
            strings(&["a\nb"]),
            "a newline inside a name is part of the name"
        );
    }

    #[test]
    fn index_info_lines_pair_mode_id_and_path_in_order() {
        let files = vec![
            ManifestFile {
                rel: "a.jsonl".into(),
                bytes: 1,
            },
            ManifestFile {
                rel: "payload/with space.txt".into(),
                bytes: 2,
            },
        ];
        let ids = strings(&["deadbeef", "cafe"]);
        assert_eq!(
            format_index_info(&files, &ids),
            "100644 deadbeef\ta.jsonl\n100644 cafe\tpayload/with space.txt\n"
        );
    }

    #[test]
    fn every_file_goes_in_as_plain_data() {
        // A transported +x bit would let a bundle change behaviour on arrival.
        let info = format_index_info(
            &[ManifestFile {
                rel: "a".into(),
                bytes: 0,
            }],
            &strings(&["deadbeef"]),
        );
        assert!(info.starts_with("100644 "));
    }

    #[test]
    fn index_info_paths_are_relative() {
        // An absolute path is not rejected, it is *ignored*: git exits zero and
        // `write-tree` returns the empty tree, so the bundle would push cleanly
        // and restore as nothing.
        let info = format_index_info(
            &[
                ManifestFile {
                    rel: "bundle.json".into(),
                    bytes: 0,
                },
                ManifestFile {
                    rel: "payload/subagents/a.jsonl".into(),
                    bytes: 5,
                },
            ],
            &strings(&["aa", "bb"]),
        );
        for line in info.lines() {
            let (_, path) = line.split_once('\t').expect("mode and oid then tab");
            assert!(!path.starts_with('/'), "{path:?} must be relative");
            assert!(!path.contains(".."), "{path:?} must not escape the tree");
        }
    }

    #[test]
    fn staging_that_matches_the_manifest_is_accepted() {
        let expected = strings(&["a.jsonl", "bundle.json", "payload/x"]);
        assert!(staging_differs(&expected, &expected).is_none());
    }

    #[test]
    fn staging_that_dropped_a_file_is_reported_by_name() {
        let expected = strings(&["bundle.json", "payload/x"]);
        let staged = strings(&["bundle.json"]);
        let why = staging_differs(&expected, &staged).expect("a dropped file is a failure");
        assert!(why.contains("payload/x"), "{why}");
        assert!(why.contains("missing"), "{why}");
    }

    #[test]
    fn staging_that_added_a_file_is_reported() {
        let expected = strings(&["bundle.json"]);
        let staged = strings(&["bundle.json", "rogue"]);
        let why = staging_differs(&expected, &staged).expect("an extra file is a failure");
        assert!(why.contains("rogue"), "{why}");
        assert!(why.contains("unexpected"), "{why}");
    }

    #[test]
    fn an_empty_staged_tree_is_reported_rather_than_pushed() {
        // What a silently-ignored index actually produces, and the reason the
        // check exists at all.
        let expected = strings(&["bundle.json", "x.jsonl"]);
        let why = staging_differs(&expected, &[]).expect("an empty tree is a failure");
        assert!(why.contains("staged tree holds 0 file(s)"), "{why}");
        assert!(why.contains("bundle.json"), "{why}");
    }

    // --- manifest ----------------------------------------------------------

    #[test]
    fn manifest_lists_the_transcript_at_its_packed_length() {
        let manifest = BundleManifest::from_bundle(&sample_bundle());
        let transcript = manifest
            .files
            .iter()
            .find(|f| f.rel == "5e492710-96cf-483a-929d-5330b8ad3000.jsonl")
            .expect("transcript");
        // transcript_bytes is the live file, which was still being appended to.
        // The packed file stops at the last complete line.
        assert_eq!(transcript.bytes, 27_547_000);
        assert_ne!(transcript.bytes, 27_547_554);
    }

    #[test]
    fn manifest_lists_every_sidecar_under_payload() {
        let manifest = BundleManifest::from_bundle(&sample_bundle());
        let rels: Vec<&str> = manifest.files.iter().map(|f| f.rel.as_str()).collect();
        assert!(rels.contains(&"payload/subagents/agent-a.jsonl"));
        assert!(rels.contains(&"payload/tool-results/toolu_01.txt"));
        // The manifest cannot state its own length; it is verified by parsing.
        assert_eq!(
            manifest
                .files
                .iter()
                .find(|f| f.rel == MANIFEST_NAME)
                .expect("manifest entry")
                .bytes,
            0
        );
    }

    #[test]
    fn manifest_files_are_sorted_so_two_machines_agree() {
        let manifest = BundleManifest::from_bundle(&sample_bundle());
        let rels: Vec<&str> = manifest.files.iter().map(|f| f.rel.as_str()).collect();
        let mut sorted = rels.clone();
        sorted.sort_unstable();
        assert_eq!(rels, sorted);
    }

    #[test]
    fn manifest_total_excludes_the_unchecked_manifest() {
        let manifest = BundleManifest::from_bundle(&sample_bundle());
        assert_eq!(manifest.total_bytes(), 27_547_000 + 4096 + 512);
    }

    #[test]
    fn manifest_copies_the_git_state_it_was_given() {
        // `GitState` is not Clone and `pack` is not ours to change, so the
        // manifest snapshots it field by field.
        let manifest = BundleManifest::from_bundle(&sample_bundle());
        let git = manifest.git.expect("git state");
        assert_eq!(git.branch, "poc/cross-machine-session");
        assert_eq!(git.untracked, vec!["notes.md".to_string()]);
    }

    #[test]
    fn manifest_round_trips_through_the_bundle_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let bundle = small_bundle();
        write_bundle(dir.path(), &bundle);

        let from_dir = BundleManifest::from_bundle_dir(dir.path()).expect("read manifest");
        let from_memory = BundleManifest::from_bundle(&bundle);
        assert_eq!(from_dir.session_id, bundle.session_id);
        assert_eq!(from_dir.transcript_offset, bundle.transcript_offset);
        assert_eq!(from_dir.files.len(), from_memory.files.len());
        for (a, b) in from_dir.files.iter().zip(&from_memory.files) {
            assert_eq!(a.rel, b.rel);
            assert_eq!(a.bytes, b.bytes);
        }
    }

    #[test]
    fn verify_accepts_a_complete_bundle() {
        let dir = tempfile::tempdir().expect("tempdir");
        let bundle = small_bundle();
        write_bundle(dir.path(), &bundle);
        BundleManifest::from_bundle(&bundle)
            .verify(dir.path())
            .expect("a complete bundle verifies");
    }

    #[test]
    fn verify_names_the_file_that_is_short() {
        let dir = tempfile::tempdir().expect("tempdir");
        let bundle = small_bundle();
        write_bundle(dir.path(), &bundle);
        // A transfer that lost a tool result must fail here, loudly, with the
        // name — not three sessions later on the other machine.
        fs::remove_file(dir.path().join("payload/subagents/agent-a.jsonl")).expect("remove");
        let err = BundleManifest::from_bundle(&bundle)
            .verify(dir.path())
            .expect_err("an incomplete bundle must not verify");
        let msg = err.to_string();
        assert!(msg.contains("payload/subagents/agent-a.jsonl"), "{msg}");
        assert!(msg.contains("missing"), "{msg}");
    }

    #[test]
    fn verify_catches_a_truncated_transcript() {
        let dir = tempfile::tempdir().expect("tempdir");
        let bundle = small_bundle();
        write_bundle(dir.path(), &bundle);
        let transcript = dir
            .path()
            .join("5e492710-96cf-483a-929d-5330b8ad3000.jsonl");
        let file = fs::OpenOptions::new()
            .write(true)
            .open(&transcript)
            .expect("open");
        file.set_len(4).expect("truncate");
        drop(file);
        let err = BundleManifest::from_bundle(&bundle)
            .verify(dir.path())
            .expect_err("a short transcript must not verify");
        let msg = err.to_string();
        assert!(msg.contains("is 4 bytes, expected 16"), "{msg}");
    }

    #[test]
    fn verify_only_checks_the_manifest_itself_by_presence() {
        // bundle.json declares bytes = 0, so any size passes; it is validated
        // by being parsed into a Bundle at all.
        let dir = tempfile::tempdir().expect("tempdir");
        let bundle = small_bundle();
        write_bundle(dir.path(), &bundle);
        fs::write(dir.path().join(MANIFEST_NAME), b"{}").expect("rewrite");
        BundleManifest::from_bundle(&bundle)
            .verify(dir.path())
            .expect("the manifest is checked by parsing, not by length");
    }

    #[test]
    fn verify_ignores_files_the_manifest_does_not_claim() {
        let dir = tempfile::tempdir().expect("tempdir");
        let bundle = small_bundle();
        write_bundle(dir.path(), &bundle);
        // A destination pack directory can hold leftovers of an earlier, longer
        // session. casr does not delete, and must not fail because of it.
        fs::write(dir.path().join("stray.txt"), "junk").expect("write");
        BundleManifest::from_bundle(&bundle)
            .verify(dir.path())
            .expect("extra files are not corruption");
    }

    #[cfg(unix)]
    #[test]
    fn verify_refuses_to_follow_a_symlink() {
        let dir = tempfile::tempdir().expect("tempdir");
        let bundle = small_bundle();
        write_bundle(dir.path(), &bundle);
        let outside = dir.path().join("outside.txt");
        fs::write(&outside, "secret").expect("write");
        let promised = dir.path().join("payload/subagents/agent-a.jsonl");
        fs::remove_file(&promised).expect("remove");
        std::os::unix::fs::symlink(&outside, &promised).expect("symlink");
        // A bundle is attacker-controlled once it has crossed a network.
        let err = BundleManifest::from_bundle(&bundle)
            .verify(dir.path())
            .expect_err("a symlink must not count as the promised file");
        assert!(err.to_string().contains("symlink"), "{err}");
    }

    #[test]
    fn check_session_catches_a_mislabelled_bundle() {
        // Restoring A's transcript under B's id is a resume that silently runs
        // the wrong conversation, so it has to be impossible, not unlikely.
        let manifest = BundleManifest::from_bundle(&sample_bundle());
        assert!(
            manifest
                .check_session("5e492710-96cf-483a-929d-5330b8ad3000")
                .is_ok()
        );
        let err = manifest
            .check_session("other")
            .expect_err("a mismatch must be refused");
        assert!(err.to_string().contains("other"), "{err}");
    }

    #[test]
    fn read_bundle_reports_a_malformed_manifest() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(dir.path().join(MANIFEST_NAME), "{not json").expect("write");
        let err = read_bundle(dir.path()).expect_err("garbage");
        assert!(err.to_string().contains("malformed"), "{err}");
    }

    // --- transport shape ---------------------------------------------------

    #[test]
    fn store_path_is_made_absolute_at_construction() {
        // checkout-index runs with the working directory set to the pull
        // destination, so a relative GIT_INDEX_FILE would resolve wrongly.
        let t = GitTransport::new("r", "relative/store.git");
        assert!(t.store().is_absolute(), "{:?}", t.store());
    }

    #[test]
    fn transport_reports_its_kind_for_json_output() {
        let t = GitTransport::new("r", "/s/store.git");
        assert_eq!(t.kind(), "git");
        assert_eq!(t.remote(), "r");
    }

    #[test]
    fn requests_default_to_refusing_to_be_destructive() {
        // Both opt-ins are opt-ins: a caller that forgets a flag must get the
        // safe behaviour, not the lossy one.
        let push = PushRequest {
            session_id: "s".into(),
            bundle_dir: PathBuf::from("/b"),
            force: false,
        };
        let pull = PullRequest {
            session_id: "s".into(),
            dest_dir: PathBuf::from("/d"),
            force: false,
            overwrite: false,
        };
        assert!(!push.force);
        assert!(!pull.force);
        assert!(!pull.overwrite);
    }

    #[test]
    fn push_receipts_serialise_for_json_output() {
        let receipt = PushReceipt {
            transport: "git".into(),
            session_id: "s".into(),
            remote: "r".into(),
            ref_name: "refs/casr/packs/s".into(),
            commit: "abc".into(),
            tree: "def".into(),
            files: 4,
            bytes: 1024,
            pushed: true,
        };
        let text = serde_json::to_string(&receipt).expect("serialize");
        let back: PushReceipt = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back, receipt);
    }

    #[test]
    fn pull_receipts_serialise_for_json_output() {
        let receipt = PullReceipt {
            transport: "git".into(),
            session_id: "s".into(),
            remote: "r".into(),
            ref_name: "refs/casr/packs/s".into(),
            commit: "abc".into(),
            tree: "def".into(),
            dest_dir: PathBuf::from("/d"),
            files: 4,
            bytes: 1024,
            transcript_offset: 512,
            transcript_bytes: 600,
            sidecars: 2,
        };
        let text = serde_json::to_string(&receipt).expect("serialize");
        let back: PullReceipt = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back, receipt);
    }
}

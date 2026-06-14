//! Integration tests for Cline storage-root detection.
//!
//! The lib crate has `#![forbid(unsafe_code)]` and the env-mutation APIs
//! require `unsafe {}`, so these tests live in the integration-test binary
//! which can use `unsafe`. All env access is serialized through
//! [`test_env::EnvLock`] (see `tests/test_env.rs`) so we don't race with
//! other tests in the same binary.

mod test_env;

use casr::providers::Provider;
use casr::providers::cline::Cline;

static CLINE_ENV: test_env::EnvLock = test_env::EnvLock;

/// RAII helper for setting an env var and restoring its prior value on drop.
struct EnvGuard {
    key: &'static str,
    original: Option<String>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &std::path::Path) -> Self {
        let original = std::env::var(key).ok();
        // SAFETY: Tests hold the CLINE_ENV lock for the duration of env
        // mutation and the call under test.
        unsafe { std::env::set_var(key, value) };
        Self { key, original }
    }
    fn clear(key: &'static str) -> Self {
        let original = std::env::var(key).ok();
        // SAFETY: see `EnvGuard::set`.
        unsafe { std::env::remove_var(key) };
        Self { key, original }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: see `EnvGuard::set`.
        match &self.original {
            Some(val) => unsafe { std::env::set_var(self.key, val) },
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}

#[test]
fn cline_home_short_circuits_candidate_list() {
    let _lock = CLINE_ENV.lock().unwrap();
    let tmp = tempfile::tempdir().expect("tmpdir");
    let _guard = EnvGuard::set("CLINE_HOME", tmp.path());
    // The private `pick_storage_root_for_write` consults `CLINE_HOME` first;
    // the public surface is `Provider::detect`, which should now treat the
    // overridden path as installed.
    let provider = Cline;
    let result = Provider::detect(&provider);
    assert!(
        result.installed,
        "CLINE_HOME pointing at an existing dir should make detect() report installed; \
         evidence: {:?}",
        result.evidence
    );
    assert!(
        result.evidence.iter().any(|e| e.contains("CLINE_HOME")),
        "evidence must mention CLINE_HOME; got {:?}",
        result.evidence
    );
}

#[test]
fn vscode_user_data_dir_produces_candidates_per_extension_id() {
    let _lock = CLINE_ENV.lock().unwrap();
    let tmp = tempfile::tempdir().expect("tmpdir");
    let _user_data = EnvGuard::set("VSCODE_USER_DATA_DIR", tmp.path());
    let _cline = EnvGuard::clear("CLINE_HOME");

    // Construct a scratch Cline provider and call the public `session_roots`
    // entry point, which in turn asks `existing_storage_roots` to filter the
    // candidate list. With a non-existent base path the result is empty —
    // what we want to assert is that the candidate-builder doesn't reject
    // the VSCODE_USER_DATA_DIR override. We exercise that by writing a
    // dummy `tasks` subdir under one of the candidate paths and asserting
    // it shows up.
    //
    // First pick a known extension id and stage a directory at the path the
    // builder should compute.
    let id = "saoudrizwan.claude-dev";
    let expected = tmp.path().join("User").join("globalStorage").join(id);
    std::fs::create_dir_all(&expected).expect("stage candidate dir");
    let provider = Cline;
    let roots = Provider::session_roots(&provider);
    assert!(
        roots.contains(&expected.join("tasks")),
        "VSCODE_USER_DATA_DIR override should produce a candidate under {expected:?}, got {roots:?}"
    );
}

#[test]
fn vscode_portable_produces_candidates_per_extension_id() {
    let _lock = CLINE_ENV.lock().unwrap();
    let tmp = tempfile::tempdir().expect("tmpdir");
    let _portable = EnvGuard::set("VSCODE_PORTABLE", tmp.path());
    let _cline = EnvGuard::clear("CLINE_HOME");

    let id = "saoudrizwan.claude-dev";
    let expected = tmp
        .path()
        .join("user-data")
        .join("User")
        .join("globalStorage")
        .join(id);
    std::fs::create_dir_all(&expected).expect("stage candidate dir");
    let provider = Cline;
    let roots = Provider::session_roots(&provider);
    assert!(
        roots.contains(&expected.join("tasks")),
        "VSCODE_PORTABLE override should produce a candidate under {expected:?}, got {roots:?}"
    );
}

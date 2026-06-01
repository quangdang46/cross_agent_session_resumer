# PLAN: Unify jcode session resume/export on CASR

## Goal

1. **Unify** jcode's session import/resume on CASR (delete jcode's in-house
   importer, `jcode-import-core` + `crate::import`).
2. **CASR stays standalone** for every existing provider (`cc`, `cod`, `gmi`, …).
   The jcode work is purely additive.
3. **CASR ships a library API** that jcode links against to power:
   - `jcode resume <id-from-any-provider>`  (import: any provider → jcode)
   - `jcode export <provider>`              (export: jcode → any provider)

## Non-negotiable invariant: dependency direction

```
CASR (single crate, builds standalone, curl|bash distributable)
   ▲
   │  casr = { path = "../cross_agent_session_resumer" }   ← ONE direction only
   │
jcode workspace (jcode-app-core / jcode binary)  → depends on casr
```

- **CASR depends on NOTHING in jcode.** The jcode provider inside CASR is a
  self-contained format parser/writer with its own `serde` structs.
- **jcode depends on CASR.** One direction → no cycle, and CASR keeps building
  outside the jcode workspace (its distribution model requires this).

> ⚠️ The previous draft showed CASR's jcode provider calling
> `Session::load_from_path()`. That type lives in `jcode-base` (a heavy internal
> jcode crate) and its journal types are `pub(super)`. Depending on it would
> break CASR's standalone build. **Do not depend on any jcode crate.**

---

## PHASE 1 — CASR: self-contained `jcode` provider

### 1.1 Files to modify (CASR)

| File | Change |
|---|---|
| `src/providers/jcode.rs` | **NEW** — provider impl + local serde structs + tests |
| `src/providers/mod.rs` | `pub mod jcode;` |
| `src/discovery.rs` | add `Box::new(jcode::JCode)` to `default_registry()`; add a `.json` jcode signature branch in `infer_provider_for_path` (for `--source <path>` import) |
| `src/main.rs` | add `"-jc" => Some("jc")` to `rewrite_shorthand_resume_args` (shorthand is a hardcoded match; registration alone does NOT enable `casr -jc`) |

`casr jc resume <id>` and `--source jc` work automatically once registered —
`find_by_alias` matches slug **or** cli_alias with `-`/`_`/space normalization.
`jc` / `jcode` do not collide with existing aliases
(`cc cod gmi cur cln aid amp opc gpt cwb vib fac ocl pi`).

### 1.2 Provider trait — actual signatures (from `src/providers/mod.rs`)

```rust
pub struct JCode;

impl Provider for JCode {
    fn name(&self) -> &str { "jcode" }
    fn slug(&self) -> &str { "jcode" }
    fn cli_alias(&self) -> &str { "jc" }

    fn detect(&self) -> DetectionResult;                 // `which jcode` + sessions dir
    fn session_roots(&self) -> Vec<PathBuf>;             // jcode_dir()/sessions
    fn owns_session(&self, id: &str) -> Option<PathBuf>; // <root>/<id>.json
    fn read_session(&self, path: &Path) -> anyhow::Result<CanonicalSession>;
    fn write_session(&self, s: &CanonicalSession, o: &WriteOptions)
        -> anyhow::Result<WrittenSession>;
    fn resume_command(&self, id: &str) -> String;        // "jcode --resume <id>"
    fn list_sessions(&self) -> Option<Vec<(String, PathBuf)>>; // glob sessions/*.json
}
```

`WrittenSession { paths, session_id, resume_command, backup_path }`.

### 1.3 Storage path resolution (replicate `jcode_dir()` precedence exactly)

From `jcode-storage`:

| Condition | Sessions dir |
|---|---|
| `$JCODE_HOME` set | `$JCODE_HOME/sessions/` |
| `$JCODE_USE_XDG` truthy | `${XDG_DATA_HOME:-$HOME/.local/share}/jcode/sessions/` |
| default | `$HOME/.jcode/sessions/` |

Truthy = `1|true|yes|on` (case-insensitive). Precedence: `JCODE_HOME` >
`JCODE_USE_XDG`/XDG > `~/.jcode`.

### 1.4 On-disk format (confirmed in `jcode-base/session/storage_paths.rs`)

```
<sessions>/<id>.json            ← snapshot (full Session, serde)
<sessions>/<id>.journal.jsonl   ← append-only log; journal name = <stem>.journal.jsonl
```

`read_session(<id>.json)` must: parse the snapshot, then if
`<id>.journal.jsonl` exists, apply each line (a `SessionJournalEntry`) by
extending `messages` / `env_snapshots` / `memory_injections` / `replay_events`
and overwriting metadata from `meta`. **Reimplement this locally** (CASR cannot
use jcode's private journal types).

### 1.5 Local serde structs (CASR-owned, minimal subset)

Mirror only what we read/write. jcode's real types live in
`jcode-message-types` / `jcode-session-types` / `jcode-base` — **do not import
them**; copy the wire shapes:

```rust
// Role is lowercase and has ONLY two variants in jcode.
#[derive(Serialize, Deserialize)] #[serde(rename_all = "lowercase")]
enum JRole { User, Assistant }

#[derive(Serialize, Deserialize)] #[serde(tag = "type", rename_all = "snake_case")]
enum JBlock {
    Text { text: String, #[serde(skip_serializing_if="Option::is_none")] cache_control: Option<serde_json::Value> },
    Reasoning { text: String },
    AnthropicThinking { thinking: String, signature: String },
    OpenAIReasoning { id: String, summary: Vec<String>, /* + optional fields */ },
    ToolUse { id: String, name: String, input: serde_json::Value },
    ToolResult { tool_use_id: String, content: String, #[serde(skip_serializing_if="Option::is_none")] is_error: Option<bool> },
    Image { media_type: String, data: String },
    OpenAICompaction { encrypted_content: String },
}

#[derive(Serialize, Deserialize)]
struct JStoredMessage {
    id: String,                 // REQUIRED (not optional)
    role: JRole,
    content: Vec<JBlock>,
    #[serde(default, skip_serializing_if="Option::is_none")] timestamp: Option<String>, // RFC3339
    // display_role / tool_duration_ms / token_usage: preserve via extra if needed
}

#[derive(Serialize, Deserialize)]
struct JSnapshot {
    id: String,
    #[serde(default)] parent_id: Option<String>,
    #[serde(default)] title: Option<String>,
    created_at: String, updated_at: String,           // RFC3339, REQUIRED
    messages: Vec<JStoredMessage>,
    #[serde(default)] provider_session_id: Option<String>,
    #[serde(default)] provider_key: Option<String>,
    #[serde(default)] model: Option<String>,
    #[serde(default)] working_dir: Option<String>,
    #[serde(default)] status: serde_json::Value,       // default to "Active" on write
    // env_snapshots / memory_injections / replay_events / compaction …
    //   → round-trip through CanonicalSession.metadata.jcode
}
```

### 1.6 ContentBlock → CanonicalMessage mapping

| jcode block | → canonical |
|---|---|
| `Text { text }` | append `text` to `content` |
| `ToolUse { id, name, input }` | `tool_calls.push(ToolCall { id: Some(id), name, arguments: input })` |
| `ToolResult { tool_use_id, content, is_error }` | `tool_results.push(ToolResult { call_id: Some(tool_use_id), content, is_error })` |
| `Reasoning` / `AnthropicThinking` / `OpenAIReasoning` | author=`"reasoning"`; preserve in `extra` (dropped on cross-agent handoff unless `--keep-reasoning`) |
| `Image` | `[Image: <media_type>]`; full data preserved in `extra` |
| `OpenAICompaction` | **skip on read AND write** (runtime-only; jcode re-compacts on resume) |

### 1.7 jcode role model (read/write asymmetry — important)

jcode `Role` = `{ User, Assistant }` only (no System/Tool). jcode stores tool
results as `Role::User` messages containing a `ToolResult` block.

- **Read:** a User message whose blocks are tool results → canonical role `Tool`
  with `tool_results`; otherwise `User`/`Assistant` as-is.
- **Write:** collapse canonical `System` / `Tool` / `Other` → `User`. CASR's
  read-back role buckets already fold `User|System|Tool|Other → "user"`, so
  buckets match as long as the writer is consistent.

### 1.8 Round-trip / read-back correctness (the real work)

`pipeline::convert` re-reads the written jcode file and compares **count + role
bucket + exact content bytes** to the canonical. The reader and writer must be
exact inverses:

- Pipeline **step 7b fires for jcode** (non-Claude target): messages with empty
  `content` but tool calls/results get synthesized content
  (`[Tool: name]`, `[Tool Output] …`, `[Tool Error] …`) **before** write. The
  writer must persist that exact `content`; the reader must read it back
  identically — else "wrote N bytes, read back M bytes".
- Safest writer: emit the message body as a single `Text { text: content }`
  block, plus reconstructed `ToolUse`/`ToolResult` blocks from
  `tool_calls`/`tool_results`. The reader's text-merge then reproduces
  `content` verbatim.
- Reasoning/Thinking/Image string transforms must be deterministic (read-back
  re-runs the same reader, so determinism = equality). Prefer keeping rich
  payloads in `extra` over emitting lossy blocks on cross-provider import.

### 1.9 Misc

- Timestamps: canonical = epoch millis; jcode = RFC3339 `DateTime<Utc>`.
  `parse_timestamp` handles RFC3339 on read; format millis → RFC3339 on write.
- `JStoredMessage.id`, `created_at`, `updated_at`, `status` are required on
  write — synthesize stable ids; default `status="Active"`.

---

## PHASE 2 — CASR library API (already public)

`src/lib.rs` already exposes: `pipeline`, `discovery`, `model`, `providers`,
`responses`, `error`. jcode constructs the pipeline directly:

```rust
let pipeline = casr::pipeline::ConversionPipeline {
    registry: casr::discovery::ProviderRegistry::default_registry(),
};
let res = pipeline.convert(target_alias, source_id, casr::pipeline::ConvertOptions::default())?;
// res.written: Option<WrittenSession { paths, session_id, resume_command, backup_path }>
```

`ConvertOptions` fields (ALL required unless using `..Default::default()`):
`dry_run, force, verbose, enrich, source_hint, max_context_tokens,
max_tool_output, keep_reasoning`.
> The earlier draft omitted `verbose` → would not compile. Use
> `..Default::default()` and override only what you need.

**Optional polish:** add two thin wrappers in `lib.rs`
(`resume_into(target, id, opts)`, `export_to(provider, id, opts)`) so jcode
doesn't hand-build the pipeline. Not required for correctness.

---

## PHASE 3 — jcode `resume` (import): point the seam at CASR

The entire in-house import path funnels through ONE function
(`src/cli/dispatch.rs`):

```rust
fn resolve_resume_id(resume_id: &str) -> Result<String> {
    match session::find_session_by_name_or_id(resume_id) {
        Ok(full_id) => Ok(full_id),                                   // native jcode
        Err(native_err) => match crate::import::import_external_resume_id(resume_id)? {
            Some(imported_id) => Ok(imported_id),                     // ← swap to CASR
            None => Err(native_err),
        },
    }
}
```

Replacement (keep **native-first, CASR-fallback** ordering — avoids
re-converting jcode's own ids and is faster):

```rust
// crates/jcode-app-core/src/agent/casr_integration.rs  (NEW)
pub fn casr_import_to_jcode(source_id: &str) -> anyhow::Result<Option<String>> {
    let pipeline = casr::pipeline::ConversionPipeline {
        registry: casr::discovery::ProviderRegistry::default_registry(),
    };
    match pipeline.convert("jcode", source_id, casr::pipeline::ConvertOptions::default()) {
        Ok(res) => Ok(res.written.map(|w| w.session_id)),  // new jcode id → existing Session::load
        Err(_)  => Ok(None),                               // fall through to native error
    }
}
```

- Add `casr` path dep to **`crates/jcode-app-core/Cargo.toml`** (the crate that
  calls it), ideally via `[workspace.dependencies] casr = { path = … }` +
  `casr.workspace = true`. (The earlier draft put the wrapper under
  `jcode-app-core/src/session/` — wrong; the `session` module lives in
  `jcode-base`.)
- `restore_session(&mut self, …) -> Result<SessionStatus>` (in
  `jcode-app-core/src/agent/turn_execution.rs`) is unchanged: it still does
  `Session::load(id)`. We only feed it the converted id. Minimal blast radius.
- **Imported sessions must start a fresh provider thread.** When CASR writes the
  jcode snapshot for a cross-provider import, set top-level
  `provider_session_id = None` and `provider_key = None` (consider
  `model = None`); keep the originals in `extra.jcode`. Otherwise
  `restore_session` emits provider-mismatch warnings and may fail model restore.
  (Mirrors `Session::fork()`, which nulls `provider_session_id`.)
- After parity + tests: `jcode-import-core` and `crate::import` become dead code.
  **Removal requires explicit owner sign-off (RULE 1) — do not delete without it.**

---

## PHASE 4 — jcode `export <provider>` (new, additive)

Today `Command::Export` only supports `ExportFormatArg { Markdown, Json, Html }`
(`src/cli/args.rs`) — NOT cross-provider. Add provider targets alongside
(extend the enum or add a `--to <alias>` arg); keep md/json/html as-is:

```rust
// export current jcode session into a target provider's native format
let res = pipeline.convert(provider_alias, &current_jcode_session_id, opts)?;
println!("{}", res.written.unwrap().resume_command); // e.g. "claude --resume <id>"
```

---

## Import / Export flow summary

```
jcode resume <codex-id>  → CASR.convert("jcode", id)        → new jcode id → Session::load → restore
jcode resume <cc-id>     → CASR.convert("jcode", id)        → new jcode id → …
jcode export cc          → CASR.convert("cc",   <jc-id>)    → claude --resume <new-id>
jcode export cod         → CASR.convert("cod",  <jc-id>)    → codex resume <new-id>

casr jc resume <any-id>  (standalone)   |  casr cc resume <jc-id>  (standalone)
casr -jc <any-id>        (after main.rs shorthand)
```

---

## Test plan (per AGENTS.md provider checklist)

- `src/providers/jcode.rs`: reader + writer unit tests against fixtures.
- `tests/fixtures/`: jcode snapshot (+journal) sample and expected canonical.
- Round-trip: `read_jcode(write_jcode(canonical)) ~= canonical`; add
  jcode rows to the cross-provider matrix (jcode↔cc, jcode↔cod, jcode↔gmi).
- CLI integration: `casr list --provider jc`, `casr info <jc-id>`,
  `casr cc resume <jc-id>`, `casr jc resume <cc-id>`, `casr -jc <id>`.
- jcode side: import seam parity test (foreign id → jcode), export-to-provider
  test. Gate `jcode-import-core` removal on these passing.
- Verify: `cargo check --all-targets`, `cargo clippy --all-targets -- -D warnings`,
  `cargo fmt --check`, `cargo test`.

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| CASR depending on jcode crates breaks standalone build / cycle | Self-contained local serde structs; zero jcode deps in CASR |
| Read-back verify fails on content mismatch | Writer/reader exact inverses; honor step-7b synthesis; deterministic transforms |
| jcode 2-role model loses System/Tool | Collapse to User on write; rely on CASR role buckets for verify |
| Imported session mis-fires `restore_session` | Null `provider_session_id`/`provider_key`/`model`; keep in `extra.jcode` |
| `OpenAICompaction` / journal replay complexity | Skip compaction blocks; read latest snapshot+journal, no incremental replay |
| Deleting in-house importer prematurely | Gate on parity tests + explicit owner sign-off (RULE 1) |

## Implementation order

```
1. CASR src/providers/jcode.rs (read+write+tests) + mod.rs + discovery.rs + main.rs
   → cargo test providers; casr list --provider jc; casr cc resume <jc-id>
2. (optional) lib.rs convenience wrappers
3. jcode: add casr dep to jcode-app-core; add casr_integration.rs
   → swap resolve_resume_id fallback to casr_import_to_jcode; parity test
4. jcode: extend Export with provider targets; export test
5. After parity: propose removing jcode-import-core + crate::import (await sign-off)
```

---
name: casr-testing
description: >-
  End-to-end smoke test for casr (Cross Agent Session Resumer) on the local
  machine. Real install of the casr binary, real conversion of an omp, pi-agent,
  or claude-code session, and real non-interactive resume + chat through Claude
  Code (`-p`/`--print`), Codex (`exec resume`), OpenCode (`run`), omp (`-p`),
  and pi (`-p`). Use when the user says "casr test", "casr smoke", "verify
  casr install", "real resume test", "test conversion to cc/cod/opc/omp/pi",
  or "check the casr pipeline end-to-end on my machine".
---

# casr-testing — end-to-end smoke test

Validate a real casr install by:

1. Building & installing the binary locally (or via `install.sh`).
2. Picking a source session from a project the user actually has on
   disk — any installed provider can be the source: `cc`, `cod`, `opc`,
   `omp`, `pi`, `gmi`, `her`, `kr`, etc.
3. Converting it to all relevant target providers.
4. Resume + chat for every installed CLI on PATH (Claude Code, Codex,
   OpenCode, omp, pi). Each gets a non-interactive one-turn call.
5. Distinguishing casr bugs from backend/CLI bugs in the failures.

## When to use

- After pulling new commits to the casr repo and you want to confirm
  conversions still work end-to-end.
- After editing any provider in `src/providers/*.rs` and you want a
  real-model smoke test, not just `cargo test`.
- When the user reports "casr is broken" — first reproduce with this
  skill before assuming.
- When validating a fresh install of casr on a new machine.

## Inputs (ask the user if missing)

1. **Source session ID** — any session UUID/id from any installed
   provider. Run from the **project workspace** (not the casr repo):
   ```bash
   cd ~/Projects/<user-project>     # NOT the casr repo
   casr list --json --limit 5       # cwd-scoped
   casr list --all --json --limit 5 # all workspaces (may be large)
   ```
   Confirm the source session appears. `list` is cwd-scoped by
   default — pass `--all` to search every workspace (slow on large
   installs). If the session is named (e.g. omp's `IssueScout.jsonl`),
   its id comes from the `session` header, not the filename.

2. **Which targets to convert and chat-test** — by default, convert to
   all providers that are installed; chat-test only those that have a
   real CLI on PATH. On this machine that typically means:
   - **cc** (claude -p), **cod** (codex exec resume),
     **opc** (opencode run), **omp** (omp --session -p),
     **pi** (pi --session -p).
   - Skip write-only providers (gmi, cur, cln, aid, amp, gpt, her,
     kr, grk) unless the user explicitly asks.

## Workflow

### Step 1 — install casr locally

#### Option A: `cargo install` (development builds)

```bash
cd /Users/tranquangdang21/Projects/cross_agent_session_resumer
cargo build --release
cargo install --path . --force
cp target/release/casr ~/.local/bin/casr   # keep in sync
~/.local/bin/casr --version               # should match git log -1 --format=%h
```

#### Option B: `install.sh` (release builds — test without touching live)

```bash
bash install.sh --dest /tmp/casr-install-test --yes --quiet
/tmp/casr-install-test/casr --version     # confirms download + verify
```

Then restore live install:
```bash
cp target/release/casr ~/.local/bin/casr
```

Always verify both paths agree:
```bash
which -a casr
~/.cargo/bin/casr --version
~/.local/bin/casr --version
```

### Step 2 — discover a real source session

From the **project workspace**, not from the casr repo:

```bash
cd ~/Projects/<user-project>          # NOT the casr repo
~/.local/bin/casr list --json --limit 5       # cwd-scoped (fast)
# OR search all workspaces (slower, comprehensive):
~/.local/bin/casr list --all --json --limit 10
```

Confirm the source session appears. Pick the provider with the most
entries or the one the user cares about most (omp/pi/cc/cod/opc).

### Step 3 — convert to relevant targets

```bash
# Core 5: cc, cod, opc, omp, pi — always include when installed
for t in cc cod opc omp pi; do
  echo "=== $t ==="
  ~/.local/bin/casr resume "$t" "$SESSION_ID" --source "$SOURCE_SLUG" --force --json \
    2>&1 | jq -c '{ok, tid: .target_session_id, cmd: .resume_command, warnings: (.warnings|length)}'
  echo
done
```

A successful convert reports `"ok": true`, a `written_paths` list,
a `resume_command`, and (often) a `warnings` array noting truncated
tool results and dropped older turns (context budget ~200K tokens).

A failure falls into these buckets:

- `SessionConflict` → re-run with `--force` (already passed).
- `VerifyFailed` → read-back mismatch; the file was rolled back. This
  is a casr bug — file an issue with the `detail` field.
- `ProviderNotInstalled` → the *target* CLI is not on PATH. Skip it
  (e.g. `gpt` and `amp` on most machines).
- `SessionNotFound` with "0 sessions scanned" → the source provider
  can't find the session id by filename. For omp/pi this may mean the
  session was written with `--name` (a named session): the id lives
  only in the file's `session` header line, not in the filename. Use
  `--source <provider>` to hint which store to search.

### Step 4 — real chat with installed CLIs

Each CLI has its own non-interactive flag. Pick the ones whose CLIs
are on PATH and run one turn each. Expect 15–90 s per call.

#### Claude Code — `claude -p`

```bash
cd ~/Projects/<user-project>
# The CC session lives under ~/.claude/projects/<dir-key>/; claude -p
# finds it from the cwd bucket.
claude -p "what was the last thing we were working on? Reply in ONE short sentence." \
  --resume "$CC_RESUME_ID" \
  --dangerously-skip-permissions \
  --output-format text 2>&1 | tail -5
```

If you see `Session model X could not be restored — using the default
model instead`, that is **expected** when converting from a provider
whose model name Claude Code does not have registered; it falls back
to its default model. The conversation context is still loaded.

**Workspace contract (GH #20):** the CC writer stamps the source
session's **recorded workspace**, not the invoking cwd. When the
source session has no workspace, the writer falls back to the cwd
with a loud warning. Use `casr resume cc <id> --workspace <path>` to
override.

#### Codex — `codex exec resume`

```bash
codex exec resume "$CODEX_RESUME_ID" "what were we working on? answer in one short sentence" \
  --dangerously-bypass-approvals-and-sandbox 2>&1 | grep -v "^hook:" | tail -5
```

If the run fails with `tool result's tool id() not found (2013)`
from a custom provider, **this is a backend issue, not casr**.
Confirm by spawning a brand-new native codex session in a tmp dir.

#### OpenCode — `opencode run`

```bash
opencode run "what were we working on? answer in one short sentence" \
  -s "ses_$CODEX_RESUME_ID" 2>&1 | tail -5
```

Known failure modes:

- `Model not found: unknown/unknown` → casr writer bug (fixed commit
  `207b9b0`). Rebuild if still seeing this.
- `UnknownError` + schema rejection for missing `state.output` or
  `state.time` → the V2 tool-part writer was not emitting the full
  native state shape. Fixed: tool parts now emit input (object),
  output, metadata, title, and time{start,end}. If you still see
  this, capture the log at `~/.local/share/opencode/log/<latest>.log`.

#### omp — `omp --session -p`

```bash
omp --session "$OMP_SESSION_PATH" -p "what were we working on? answer in one short sentence." 2>&1 | tail -5
```

omp is a fork of pi-agent. Sessions live under `~/.omp/agent/sessions/`;
use the path from the `casr resume omp` output. omp supports named
sessions (`--name IssueScout` → `IssueScout.jsonl` inside the session
directory), so the filename may not contain the session id.

#### pi — `pi --session -p`

```bash
pi --session "$PI_SESSION_PATH" -p "what were we working on? answer in one short sentence." 2>&1 | tail -5
```

pi is the original Pi Agent. Identical session format to omp but
separate home directory (`~/.pi/agent/sessions/`).

### Step 5 — capture results

Build a small table:

```
| Target | Convert | Chat  | Notes                                              |
|--------|---------|-------|----------------------------------------------------|
| cc     | ✅      | ✅    | model fell back to Claude default (expected)       |
| cod    | ✅      | ✅    | context restored correctly                         |
| opc    | ✅      | ✅    | state.output + time required for V2 schema         |
| omp    | ✅      | ✅    | named sessions: header id, not filename            |
| pi     | ✅      | ✅    | separate home from omp; workspace preserved        |
```

A row is **green** if `convert` + `chat` both succeed.
A red row is only a casr bug if the same red row reproduces in the
provider's native session.

## Common pitfalls (from real sessions)

1. **Wrong cwd** — `casr list` returns 0 sessions if you run it from
   anywhere other than the project the sessions belong to. The list
   filter defaults to current working directory. Use `--all` to bypass.
2. **Shorthand flag** — `casr -cc <id>` is *not* a valid casr
   command. The shape is `casr resume cc <id>`. The `-cc` style is
   a muscle-memory carryover from a different tool.
3. **omp named sessions** — `pi -n foo` / `omp --name IssueScout` creates
   `<name>.jsonl` inside the session directory. The filename does not
   contain the session id; it lives only in the `session` header line.
   `casr list` sees it (reads headers), but `owns_session` must now
   also scan header ids for underscore-less files (fixed commit
   `4d9707f`). If you still get `SessionNotFound`, pass `--source omp`
   or `--source pi`.
4. **omp and pi are SEPARATE providers** — they share the same JSONL
   format but have different home directories and env overrides.
   - omp: `~/.omp/agent/sessions/`, env `OMP_HOME`
   - pi: `~/.pi/agent/sessions/`, env `PI_AGENT_HOME`
   Use `--source omp` or `--source pi` to disambiguate.
5. **Codex tool parts are top-level** — Codex native rollout uses
   `function_call` and `function_call_output` as their own
   `response_item` envelopes, not as `tool_use`/`tool_result` blocks
   inside a `message` envelope. The casr writer must split them.
6. **OpenCode per-project `sessions` table may not have a `model`
   column** — defensively `ALTER TABLE ADD COLUMN model` before
   writing.
7. **OpenCode V2 tool-part state shape** — OpenCode 1.x validates tool
   parts and requires `state.input` as an OBJECT (not a JSON string),
   plus `state.output`, `state.time{start,end}`. Missing keys cause
   schema rejection → `UnknownError`. Fixed commit `6a76ff2`. The
   read-side handles both string and object inputs.
8. **Tool-result fidelity** — pi-agent's `toolResult` role was
   historically parsed into a `Tool` role message with
   `tool_results: vec![]`, dropping the `toolCallId` linkage. The
   reader fix threads `toolCallId` → `ToolResult.call_id` and
   `isError` → `is_error`. Without this, every converted session had
   zero structured tool data.

## Quick one-shot script

```bash
#!/usr/bin/env bash
# scripts/casr_smoke.sh — end-to-end casr smoke test
set -euo pipefail
SESSION_ID="${1:?usage: $0 SESSION_ID WORKSPACE_DIR SOURCE_SLUG}"
WORKSPACE="${2:?usage: $0 SESSION_ID WORKSPACE_DIR SOURCE_SLUG}"
SOURCE_SLUG="${3:?usage: $0 SESSION_ID WORKSPACE_DIR SOURCE_SLUG}"
CASR="${CASR_BIN:-/Users/tranquangdang21/.local/bin/casr}"
CHAT_PROMPT="what were we working on? answer in one short sentence."

cd "$WORKSPACE"
echo "=== install check ==="
"$CASR" --version
echo
echo "=== conversion matrix ==="
for t in cc cod opc omp pi; do
  echo "--- $t ---"
  "$CASR" resume "$t" "$SESSION_ID" --source "$SOURCE_SLUG" --force --json 2>&1 \
    | jq -c '{ok, tid: .target_session_id, cmd: .resume_command}' || echo "convert failed"
done
echo
echo "=== chat: cc ==="
CC_RESUME=$("$CASR" info "$SESSION_ID" --json | jq -r '.resume_command' | awk '{print $NF}')
claude -p "$CHAT_PROMPT" --resume "$CC_RESUME" --dangerously-skip-permissions --output-format text 2>/dev/null | tail -5
echo
echo "=== chat: cod ==="
CODEX_RESUME=$("$CASR" info "$SESSION_ID" --json | jq -r '.resume_command' | awk '{print $NF}')
codex exec resume "$CODEX_RESUME" "$CHAT_PROMPT" --dangerously-bypass-approvals-and-sandbox 2>/dev/null | grep -v "^hook:" | tail -5
echo
echo "=== chat: opc ==="
opencode run "$CHAT_PROMPT" -s "ses_$CODEX_RESUME" 2>/dev/null | tail -5
echo
echo "=== chat: omp ==="
OMP_PATH=$("$CASR" resume omp "$SESSION_ID" --source "$SOURCE_SLUG" --force --json | jq -r '.written_paths[0]')
omp --session "$OMP_PATH" -p "$CHAT_PROMPT" 2>/dev/null | tail -5
echo
echo "=== chat: pi ==="
PI_PATH=$("$CASR" resume pi "$SESSION_ID" --source "$SOURCE_SLUG" --force --json | jq -r '.written_paths[0]')
pi --session "$PI_PATH" -p "$CHAT_PROMPT" 2>/dev/null | tail -5
```

## Triage cheat sheet

| Symptom                                              | Bucket        | Action |
|------------------------------------------------------|---------------|--------|
| `Session not found` for known session                | casr          | wrong cwd OR wrong CLI shape OR named session (pass --source) |
| `read-back verification failed` after convert        | casr          | file a `VerifyFailed` issue with detail |
| omp/pi share home / invisible sessions               | casr (fixed)  | omp and pi are separate providers now; check env OMP_HOME / PI_AGENT_HOME |
| codex `tool result's tool id() not found` (2013)     | backend       | confirm with native codex; ship anyway |
| opencode `Model not found: unknown/unknown`          | casr (older)  | rebuild + reinstall; verify commit ≥ 207b9b0 |
| opencode `UnknownError` + missing `state.output`     | casr (fixed)  | rebuild; fix commit 6a76ff2 emits full V2 state shape |
| opencode schema rejection `Expected object` in input | casr (fixed)  | rebuild; fix commit 6a76ff2 emits input as object, not string |
| claude `Session model X not recognized`              | casr-ok       | expected; Claude Code falls back to default |
| claude session not found from cwd                    | casr-ok       | GH #20: CC writer stamps recorded workspace; use --workspace to override |
| `agent role 'subagent' must define a description`    | harmless      | opencode CLI warning; ignored |

## Out of scope

- Authoring new providers — see the `Adding New Providers` section
  in `AGENTS.md` for the casr repo.
- Releasing a new casr version — see `## Release Process` in
  `AGENTS.md` for the dist workflow.
- Cross-platform build verification — the local install here is
  macOS aarch64 only.

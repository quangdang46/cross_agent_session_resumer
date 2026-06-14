---
name: casr-testing
description: >-
  End-to-end smoke test for casr (Cross Agent Session Resumer) on the local
  machine. Real install of the casr binary, real conversion of an omp/pi-agent
  session, and real non-interactive resume + chat through Claude Code
  (`-p`/`--print`), Codex (`exec resume`), and OpenCode (`run`). Use when
  the user says "casr test", "casr smoke", "verify casr install", "real
  resume test", "test conversion to cc/cod/opc", or "check the casr
  pipeline end-to-end on my machine".
---

# casr-testing — end-to-end smoke test

Validate a real casr install by:

1. Building & installing the binary locally.
2. Picking a source session (omp / pi-agent) from a project the user
   actually has on disk.
3. Converting it to all 9 target providers (`cc`, `cod`, `gmi`, `cur`,
   `cln`, `aid`, `amp`, `opc`, `gpt`).
4. Resume + chat for the three installs that have a real CLI on PATH
   (Claude Code, Codex, OpenCode).
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

1. **Source session ID** — the omp/pi-agent session UUIDv7
   (e.g. `019ec18b-0274-7000-9d5a-b1f2d9b51a00`) and the **workspace
   directory** it was recorded in (e.g. `~/Projects/jcode`). casr
   `list` is cwd-scoped — running it from the wrong directory silently
   reports 0 sessions.
2. **Which targets to chat-test** — by default, all three installed
   CLIs (Claude Code, Codex, OpenCode). Skip the others (Aider, Cline,
   Cursor, Amp, ChatGPT, Gemini) unless the user explicitly asks —
   they are write-only on most machines.

## Workflow

### Step 1 — install casr locally

```bash
cd /Users/tranquangdang21/Projects/cross_agent_session_resumer
cargo build --release
cargo install --path . --force
~/.cargo/bin/casr --version
```

The install puts the binary at `~/.cargo/bin/casr`. If a stale
`casr` from a previous install already lives at `~/.local/bin/casr`,
the new binary will be at `~/.cargo/bin/casr` and PATH ordering
matters — invoke by absolute path to be sure.

```bash
which -a casr
~/.cargo/bin/casr --version    # should match `git log -1 --format=%h`
```

### Step 2 — discover a real source session

From the **project workspace**, not from the casr repo:

```bash
cd ~/Projects/<user-project>     # NOT the casr repo
~/.cargo/bin/casr list --json --limit 5
```

Confirm the source session appears. If `list` is empty, you are in
the wrong cwd. The casr repo and the project with sessions are
usually different directories.

### Step 3 — convert to all 9 targets

```bash
for t in cc cod gmi cur cln aid amp opc gpt; do
  echo "=== $t ==="
  ~/.cargo/bin/casr resume "$t" "$SESSION_ID" --force --json \
    2>&1 | tail -10
  echo
done
```

A successful convert reports `"ok": true`, a `written_paths` list,
a `resume_command`, and (often) a `warnings` array noting truncated
tool results and dropped older turns (context budget ~200K tokens).

A failure falls into one of these buckets:

- `SessionConflict` → re-run with `--force` (already passed).
- `VerifyFailed` → read-back mismatch; the file was rolled back. This
  is a casr bug — file an issue with the `detail` field.
- `ProviderNotInstalled` → the *target* CLI is not on PATH. Skip it
  (e.g. `gpt` and `amp` on most machines).
- Hard `InternalError` (e.g. cline) → reader needs an env var
  (`CLINE_HOME`) that the user has not set.

### Step 4 — real chat with installed CLIs

Each CLI has its own non-interactive flag. Pick the one that prints
output and exits (so you can capture stdout, set timeouts, and
distinguish "still thinking" from "failed").

#### Claude Code — `claude -p`

```bash
~/.cargo/bin/casr resume cc "$SESSION_ID" --force --json >/dev/null
RESUME_ID=$(~/.cargo/bin/casr info "$SESSION_ID" --json \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['resume_command'].split()[-1])")

claude -p "what was the last thing we were working on? Reply in one short paragraph." \
  --resume "$RESUME_ID" \
  --dangerously-skip-permissions \
  --output-format text
```

`claude -p` is `--print`; it runs one turn and exits. Real model
output should arrive in 15–90 s depending on context. If you see
`Session model X could not be restored (not a model this version
recognizes) — using the default model instead`, that is **expected**
when converting from a provider whose model name Claude Code does
not have registered; it falls back to its default model.

#### Codex — `codex exec resume`

```bash
~/.cargo/bin/casr resume cod "$SESSION_ID" --force --json >/dev/null
codex exec resume "$RESUME_ID" "say hi in one word" \
  --dangerously-bypass-approvals-and-sandbox --json 2>&1 | tail -10
```

`codex exec resume <id> "prompt"` is the non-interactive shape.
`--dangerously-bypass-approvals-and-sandbox` skips Codex's
approval prompts.

If the run fails with `tool result's tool id() not found (2013)`
from a custom provider (e.g. `minimax/MiniMax-M3`), **this is a
backend issue, not casr**. Confirm by spawning a brand-new native
codex session in a tmp dir and asking it to call a tool — same 2013
error means the provider's tool-call validator is broken; ship
anyway.

#### OpenCode — `opencode run`

```bash
~/.cargo/bin/casr resume opc "$SESSION_ID" --force --json >/dev/null
opencode run "say hi" -s "ses_$RESUME_ID" --print-logs 2>&1 | tail -10
```

OpenCode's non-interactive shape is `opencode run <msg> -s <id>`.
Two failure modes to recognize:

- `Model not found: unknown/unknown` from `SessionPrompt.getModel` →
  casr writer bug (per-message and per-session `model` column was
  hardcoded to "unknown"). Fixed in casr commit `207b9b0`. If you
  still see this, the binary is older than that commit.
- `UnknownError: UnknownError` from `SessionPrompt.run` with no
  cause line → likely a mismatch between the export JSON shape
  (per-message `model: {providerID, modelID}` for user, flat
  `modelID`/`providerID` for assistant) and what the importer
  expects. Capture the opencode server log at
  `~/.local/share/opencode/log/<latest>.log` to see the missing key.

### Step 5 — capture results

Build a small table:

```
| Target | Convert | Resume | Chat  | Notes                                    |
|--------|---------|--------|-------|------------------------------------------|
| cc     | ok      | ok     | ok    | model fell back to Claude default        |
| cod    | ok      | fail   | n/a   | 2013 — backend, not casr                 |
| gmi    | ok      | -      | -     | (no smoke test if gemini CLI is absent)  |
| opc    | ok      | fail   | n/a   | UnknownError after model fix; need       |
|        |         |        |       | deeper opencode import-format work       |
```

A row is **green** if `convert` + `resume` + `chat` all succeed.
A red row is only a casr bug if the same red row reproduces in the
provider's native session.

## Common pitfalls (from real sessions)

1. **Wrong cwd** — `casr list` returns 0 sessions if you run it from
   anywhere other than the project the sessions belong to. The list
   filter is hard-coded to current working directory.
2. **Shorthand flag** — `casr -cc <id>` is *not* a valid casr
   command. The shape is `casr resume cc <id>`. The `-cc` style is
   a muscle-memory carryover from a different tool.
3. **omp session header has no `modelId`** — the model is only in
   `model_change` events under the field `model`, not `modelId`. The
   casr reader must check both. Fixed in commit `207b9b0`.
4. **Codex tool parts are top-level** — Codex native rollout uses
   `function_call` and `function_call_output` as their own
   `response_item` envelopes, not as `tool_use`/`tool_result` blocks
   inside a `message` envelope. The casr writer must split them.
5. **OpenCode per-project `sessions` table may not have a `model`
   column** — defensively `ALTER TABLE ADD COLUMN model` before
   writing.
6. **OpenCode `run` reads the running server's db** — if a long-lived
   `opencode` process is using `~/.local/share/opencode/opencode.db`
   while the converted session lives in `<workspace>/.opencode/...`,
   the server may not see it. Stop the running opencode first or
   pass `--attach http://127.0.0.1:<port>` (find the port via
   `lsof -nP -iTCP -sTCP:LISTEN | grep opencode`).
7. **Tool-result fidelity was the canonical bug** — pi-agent's
   `toolResult` role was historically parsed into a `Tool` role
   message with `tool_results: vec![]`, dropping the `toolCallId`
   linkage. The reader fix in commit `dafae3c` threads `toolCallId`
   → `ToolResult.call_id` and `isError` → `is_error`. Without this,
   every converted session had zero structured tool data and the
   target CLI could not call tools.

## Quick one-shot script

For repeat runs, save this as `scripts/casr_smoke.sh` in the casr
repo and run with the session ID and workspace as args:

```bash
#!/usr/bin/env bash
# scripts/casr_smoke.sh — end-to-end casr smoke test
set -euo pipefail
SESSION_ID="${1:?usage: $0 SESSION_ID WORKSPACE_DIR}"
WORKSPACE="${2:?usage: $0 SESSION_ID WORKSPACE_DIR}"
CASR="${CASR_BIN:-/Users/tranquangdang21/.cargo/bin/casr}"

cd "$WORKSPACE"
echo "=== install check ==="
"$CASR" --version
echo
echo "=== providers ==="
"$CASR" providers 2>&1 | head -25
echo
echo "=== conversion matrix ==="
for t in cc cod gmi cur cln aid amp opc gpt; do
  echo "--- $t ---"
  "$CASR" resume "$t" "$SESSION_ID" --force --json 2>&1 \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print('ok=', d.get('ok'), 'resume=', d.get('resume_command'))" \
    || echo "convert failed"
done
echo
echo "=== chat: cc ==="
RESUME_ID=$("$CASR" info "$SESSION_ID" --json | python3 -c "import json,sys; print(json.load(sys.stdin)['resume_command'].split()[-1])")
timeout 180 claude -p "what was the last thing we were working on? Reply in one short paragraph." \
  --resume "$RESUME_ID" --dangerously-skip-permissions --output-format text \
  2>/dev/null | tail -10
echo
echo "=== chat: opc ==="
timeout 60 opencode run "say hi" -s "ses_$RESUME_ID" --print-logs 2>&1 \
  | grep -iE "model not found|unknownerror|provider|turn" | head -5
```

The script separates install check, conversion matrix, and chat
attempts so a failure in one stage does not abort the others.

## Triage cheat sheet

| Symptom                                          | Bucket        | Action |
|--------------------------------------------------|---------------|--------|
| `Session not found` for known session            | casr          | wrong cwd OR wrong CLI shape |
| `read-back verification failed` after convert    | casr          | file a `VerifyFailed` issue with detail |
| codex `tool result's tool id() not found` (2013) | backend       | confirm with native codex; ship anyway |
| opencode `Model not found: unknown/unknown`      | casr (older)  | rebuild + reinstall; verify commit ≥ 207b9b0 |
| opencode `UnknownError: UnknownError` in `run`   | casr (open)   | capture server log; inspect export JSON shape |
| claude `Session model X not recognized`          | casr-ok       | expected; Claude Code falls back to default |
| `agent role 'subagent' must define a description`| harmless      | casr emits role "subagent" without description; ignored |

## Out of scope

- Authoring new providers — see the `Adding New Providers` section
  in `AGENTS.md` for the casr repo.
- Releasing a new casr version — see `## Release Process` in
  `AGENTS.md` for the dist workflow.
- Cross-platform build verification — the local install here is
  macOS aarch64 only.

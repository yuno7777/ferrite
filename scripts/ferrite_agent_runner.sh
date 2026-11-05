#!/usr/bin/env bash
set -euo pipefail

# Autonomous Ferrite Engineering Maintenance Runner
PROJECT_DIR="/home/abhishek/Projects/ferrite-audit"
AGENT_DIR="${PROJECT_DIR}/.agent"
LOG_FILE="${AGENT_DIR}/maintenance.log"

export PATH="/home/abhishek/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"

mkdir -p "${AGENT_DIR}"

log() {
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*" | tee -a "${LOG_FILE}"
}

log "Starting Ferrite autonomous maintenance run..."

if ! command -v codex >/dev/null 2>&1; then
    log "ERROR: codex CLI binary not found in PATH."
    exit 1
fi

cd "${PROJECT_DIR}"

log "Repository status:"
git status --short | tee -a "${LOG_FILE}"

PROMPT="CRITICAL OVERRIDE: CONTINUOUS AUTONOMOUS EXECUTION.
You are the autonomous engineer maintaining Ferrite in ${PROJECT_DIR}.
1. Inspect git status, git diff, .agent/WORK_QUEUE.md, and .agent/RECOVERY_CHECKPOINT.md.
2. Select the highest priority task with status QUEUED or IN_PROGRESS.
3. Implement the necessary code changes and tests.
4. Run 'cargo test --all-targets', 'cargo clippy --all-targets -- -D warnings', and 'cargo fmt --check'.
5. Update .agent/WORK_QUEUE.md and .agent/RECOVERY_CHECKPOINT.md.
6. Verify your changes and leave clean checkpoints."

log "Launching codex non-interactive execution..."
codex exec \
    --cd "${PROJECT_DIR}" \
    --dangerously-bypass-approvals-and-sandbox \
    "${PROMPT}" >> "${LOG_FILE}" 2>&1 || {
        log "WARNING: codex execution exited with error code $?"
        exit 1
    }

log "Ferrite autonomous maintenance run finished successfully."

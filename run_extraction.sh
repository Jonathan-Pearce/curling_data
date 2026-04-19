#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_extraction.sh
#
# Orchestrates the full curling shot-data extraction across all events in
# output/result_urls.csv, processing BATCH_SIZE events per run.
# Commands are executed inside the Docker container defined by docker-compose.yml.
#
# Usage:
#   ./run_extraction.sh              # fresh start (clears previous output)
#   ./run_extraction.sh --resume     # pick up from last checkpoint
#
# Configuration variables (override via environment):
#   BATCH_SIZE   Number of events per batch            (default: 50)
#   OUTPUT_DIR   Directory for output CSV/parquet      (default: output)
#   MIN_YEAR     Earliest event year to include        (default: 2013)
#   MAX_RETRIES  Times to retry a batch on failure     (default: 2)
#   DOCKER_SVC   Docker Compose service name           (default: curling)
# ---------------------------------------------------------------------------

set -euo pipefail

BATCH_SIZE="${BATCH_SIZE:-50}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
MIN_YEAR="${MIN_YEAR:-2013}"
MAX_RETRIES="${MAX_RETRIES:-2}"
DOCKER_SVC="${DOCKER_SVC:-curling}"
CHECKPOINT_FILE="${OUTPUT_DIR}/.checkpoint.json"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/extraction_$(date +%Y%m%d_%H%M%S).log"

# Inside the container the output directory is always /app/output (volume-mounted).
CONTAINER_OUTPUT_DIR="/app/output"
CONTAINER_CHECKPOINT="/app/output/.checkpoint.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

die() {
    log "FATAL: $*"
    exit 1
}

# Run a command inside the Docker container (no leftover container after).
docker_run() {
    docker compose run --rm "$DOCKER_SVC" "$@"
}

# Return the next_index stored in the checkpoint, or 0 if no checkpoint.
checkpoint_index() {
    if [[ -f "$CHECKPOINT_FILE" ]]; then
        python3 -c "import json,sys; d=json.load(open('$CHECKPOINT_FILE')); print(d.get('next_index',0))"
    else
        echo 0
    fi
}

# Total number of events after applying --min-year filter.
# Ask the container itself so the count matches what the extractor will process.
total_events() {
    local csv_file="${OUTPUT_DIR}/result_urls.csv"
    if [[ ! -f "$csv_file" ]]; then
        die "result_urls.csv not found at ${csv_file}. Run scrape_results.py first."
    fi
    docker compose run --rm "$DOCKER_SVC" \
        python -c "
import csv, sys
with open('/app/output/result_urls.csv') as f:
    rows = [r for r in csv.DictReader(f) if int(r.get('year') or 0) >= ${MIN_YEAR}]
print(len(rows))
" 2>/dev/null
}

# Ensure the Docker image is up to date before the first batch.
ensure_image() {
    log "Building Docker image (docker compose build) …"
    docker compose build 2>&1 | tee -a "$LOG_FILE"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

RESUME=false
if [[ "${1:-}" == "--resume" ]]; then
    RESUME=true
fi

TOTAL=$(total_events)
log "============================================================"
log "Curling extraction orchestrator"
log "  Total events : $TOTAL"
log "  Batch size   : $BATCH_SIZE"
log "  Output dir   : $OUTPUT_DIR"
log "  Min year     : $MIN_YEAR"
log "  Docker svc   : $DOCKER_SVC"
log "  Log file     : $LOG_FILE"
log "============================================================"

# Build image once upfront (no-op if already up to date).
if ! $RESUME; then
    ensure_image
fi

# On a fresh start, clear old output and checkpoint so we get a clean run.
if ! $RESUME; then
    if [[ -f "$CHECKPOINT_FILE" ]]; then
        log "Removing old checkpoint to start fresh."
        rm -f "$CHECKPOINT_FILE"
    fi
    log "Clearing previous output files (matches, ends, shots, events, teams, players)."
    for f in matches.parquet ends.parquet shot_locations_raw.parquet \
              events.parquet teams.parquet players.parquet; do
        rm -f "${OUTPUT_DIR}/${f}" "${OUTPUT_DIR}/${f}.prev"
    done
fi

BATCH_NUM=0
FIRST_BATCH=true

while true; do
    CURRENT_INDEX=$(checkpoint_index)

    if (( CURRENT_INDEX >= TOTAL )); then
        log "All $TOTAL events processed. Extraction complete."
        break
    fi

    # Safety: if the Python extractor reported "already processed" (0 remaining)
    # but the shell total doesn't match, the index won't advance — detect and stop.
    PREV_INDEX=$CURRENT_INDEX

    BATCH_NUM=$(( BATCH_NUM + 1 ))
    BATCH_END=$(( CURRENT_INDEX + BATCH_SIZE ))
    if (( BATCH_END > TOTAL )); then BATCH_END=$TOTAL; fi

    log "------------------------------------------------------------"
    log "Batch $BATCH_NUM — events $(( CURRENT_INDEX + 1 ))–${BATCH_END} of $TOTAL"
    log "------------------------------------------------------------"

    # Build the command.  First batch uses --start-event 0 (no --resume flag
    # needed); subsequent batches use --resume so they append to existing files.
    CMD=(
        python -m src.scraping.extract_shot_data
        --output-dir "$CONTAINER_OUTPUT_DIR"
        --min-year   "$MIN_YEAR"
        --batch-size "$BATCH_SIZE"
        --checkpoint "$CONTAINER_CHECKPOINT"
    )
    if ! $FIRST_BATCH || $RESUME; then
        CMD+=(--resume)
    fi

    FIRST_BATCH=false

    # Run with retry logic.
    ATTEMPT=0
    SUCCESS=false
    while (( ATTEMPT <= MAX_RETRIES )); do
        ATTEMPT=$(( ATTEMPT + 1 ))
        if (( ATTEMPT > 1 )); then
            log "  Retry $ATTEMPT/$MAX_RETRIES for batch $BATCH_NUM …"
            sleep 5
        fi

        log "  Running: docker compose run --rm $DOCKER_SVC ${CMD[*]}"
        if docker_run "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
            SUCCESS=true
            break
        else
            EXIT_CODE=$?
            log "  Batch $BATCH_NUM attempt $ATTEMPT exited with code $EXIT_CODE."
        fi
    done

    if ! $SUCCESS; then
        NEW_INDEX=$(checkpoint_index)
        if (( NEW_INDEX > CURRENT_INDEX )); then
            log "WARNING: Batch $BATCH_NUM partially succeeded (advanced from index $CURRENT_INDEX to $NEW_INDEX). Continuing."
        else
            die "Batch $BATCH_NUM failed after $MAX_RETRIES retries and made no progress. Check $LOG_FILE."
        fi
    fi

    NEW_INDEX=$(checkpoint_index)
    log "Batch $BATCH_NUM complete. Checkpoint index: ${NEW_INDEX}/${TOTAL}."

    # If the checkpoint didn't advance (extractor said "already done"), we're finished.
    if (( NEW_INDEX <= PREV_INDEX )); then
        log "No new events processed — extraction already complete at index ${NEW_INDEX}."
        break
    fi
done

log "============================================================"
log "Extraction finished successfully."
log "  Batches run : $BATCH_NUM"
log "  Output      : $OUTPUT_DIR/"
log "  Log         : $LOG_FILE"
log "============================================================"

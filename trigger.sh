#!/bin/bash
# trigger_disk_full.sh
# Simulates a "disk almost full" incident by writing dummy files
# into a target directory until usage crosses a threshold.
#
# Usage:
#   ./trigger_disk_full.sh start   -> creates dummy files
#   ./trigger_disk_full.sh status  -> shows current disk usage
#   ./trigger_disk_full.sh stop    -> removes dummy files (reset)

set -e

TARGET_DIR="/tmp/incident_sim"
DUMMY_FILE_PREFIX="dummy_bloat"
CHUNK_SIZE_MB=200      # size of each dummy file
MAX_CHUNKS=10          # safety cap so it can't actually fill a real disk
THRESHOLD_PERCENT=90   # simulated "alert" threshold

mkdir -p "$TARGET_DIR"

usage_percent() {
  df "$TARGET_DIR" | awk 'NR==2 {gsub("%","",$5); print $5}'
}

case "$1" in
  start)
    echo "[*] Starting disk-full simulation in $TARGET_DIR"
    i=0
    while [ "$(usage_percent)" -lt "$THRESHOLD_PERCENT" ] && [ "$i" -lt "$MAX_CHUNKS" ]; do
      i=$((i+1))
      FILE="$TARGET_DIR/${DUMMY_FILE_PREFIX}_${i}.bin"
      echo "  -> writing $FILE (${CHUNK_SIZE_MB}MB)"
      # fallocate is fast; falls back to dd if unavailable
      if command -v fallocate >/dev/null 2>&1; then
        fallocate -l "${CHUNK_SIZE_MB}M" "$FILE" 2>/dev/null || dd if=/dev/zero of="$FILE" bs=1M count="$CHUNK_SIZE_MB" status=none
      else
        dd if=/dev/zero of="$FILE" bs=1M count="$CHUNK_SIZE_MB" status=none
      fi
      echo "     current usage: $(usage_percent)%"
    done
    echo "[!] ALERT: disk usage at $(usage_percent)% (threshold: ${THRESHOLD_PERCENT}%)"
    echo "[!] Incident triggered. See runbook: disk_full_runbook.md"
    ;;

  status)
    echo "Current usage of $TARGET_DIR mount:"
    df -h "$TARGET_DIR"
    echo ""
    echo "Dummy files present:"
    ls -lh "$TARGET_DIR" 2>/dev/null || echo "  (none)"
    ;;

  stop)
    echo "[*] Cleaning up dummy files in $TARGET_DIR"
    rm -f "$TARGET_DIR"/${DUMMY_FILE_PREFIX}_*.bin
    echo "[*] Reset complete. Current usage:"
    df -h "$TARGET_DIR"
    ;;

  *)
    echo "Usage: $0 {start|status|stop}"
    exit 1
    ;;
esac
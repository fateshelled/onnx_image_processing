#!/bin/bash
# Download and extract TUM RGB-D freiburg2/3 sequences needed for the
# fr1-fr3 cross-dataset validation.
# Usage: download_tum_fr23.sh <state_dir>
set -u

STATE_DIR="$1"
DEST=/home/ubuntu/datasets/tum_rgbd
BASE="https://cvg.cit.tum.de/rgbd/dataset"

SEQ2="freiburg2_desk freiburg2_xyz freiburg2_rpy"
SEQ3="freiburg3_long_office_household freiburg3_sitting_xyz"

mkdir -p "$DEST"
echo $$ > "$STATE_DIR/pid"
: > "$STATE_DIR/log"
: > "$STATE_DIR/rcs"

dl() {  # dl <freiburgN> <name>
  local fr="$1" name="$2"
  local url="$BASE/$fr/rgbd_dataset_$name.tgz"
  local tgz="$DEST/rgbd_dataset_$name.tgz"
  if [ -f "$DEST/rgbd_dataset_$name/groundtruth.txt" ]; then
    echo "[$(date +%H:%M:%S)] skip $name (already extracted)" >> "$STATE_DIR/log"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] download $name" >> "$STATE_DIR/log"
  curl -L --retry 5 --retry-delay 5 -m 7200 -o "$tgz" "$url" >> "$STATE_DIR/log" 2>&1 || { echo "$name curl rc=$?" >> "$STATE_DIR/rcs"; return 1; }
  echo "[$(date +%H:%M:%S)] extract $name ($(du -h "$tgz" | cut -f1))" >> "$STATE_DIR/log"
  tar xzf "$tgz" -C "$DEST" >> "$STATE_DIR/log" 2>&1 || { echo "$name tar rc=$?" >> "$STATE_DIR/rcs"; return 1; }
  rm -f "$tgz"
  if [ -f "$DEST/rgbd_dataset_$name/groundtruth.txt" ] && [ -f "$DEST/rgbd_dataset_$name/rgb.txt" ]; then
    echo "[$(date +%H:%M:%S)] ok $name (frames=$(wc -l < "$DEST/rgbd_dataset_$name/rgb.txt"), gt=$(wc -l < "$DEST/rgbd_dataset_$name/groundtruth.txt"))" >> "$STATE_DIR/log"
  else
    echo "$name missing groundtruth.txt/rgb.txt" >> "$STATE_DIR/rcs"
  fi
}

for s in $SEQ2; do dl freiburg2 "$s"; done
for s in $SEQ3; do dl freiburg3 "$s"; done

echo "[$(date +%H:%M:%S)] DONE" >> "$STATE_DIR/log"
echo 0 > "$STATE_DIR/exit"

#!/usr/bin/env bash
set -euo pipefail

# --- Adjust if your paths differ ---
OLD="/home/vision_ai_adm/miniconda3"
NEW="/data/miniconda3"
USER_SHELL_RC="${HOME}/.bashrc"    # change to ~/.zshrc if you use zsh
MAKE_COMPAT_SYMLINK="yes"          # "yes" leaves OLD->NEW symlink (recommended)

# --- Pre-flight checks ---
if [ ! -d "$OLD" ]; then
  echo "ERROR: Old Miniconda directory not found: $OLD" >&2
  exit 1
fi

if mountpoint -q /data; then
  echo ">>> /data is a mount point."
fi

echo ">>> Checking free space on /data..."
NEEDED_KB=$(du -sk "$OLD" | awk '{print $1}')
FREE_KB=$(df -k --output=avail "/data" | tail -1 | tr -d ' ')
if (( FREE_KB <= NEEDED_KB )); then
  echo "ERROR: Not enough space on /data. Need ${NEEDED_KB} KB, have ${FREE_KB} KB." >&2
  exit 1
fi

echo ">>> Ensuring no processes are using $OLD ..."
if pgrep -f "$OLD" >/dev/null 2>&1; then
  echo "ERROR: Some processes are using $OLD. Close any Python/conda shells and try again." >&2
  exit 1
fi

# --- Deactivate conda if present ---
if command -v conda >/dev/null 2>&1; then
  echo ">>> Deactivating any active conda env..."
  # shellcheck disable=SC1090
  eval "$(conda shell.bash hook 2>/dev/null || true)"
  conda deactivate || true
fi

# --- Optional shrink: clean caches to speed copy ---
if [ -x "$OLD/bin/conda" ]; then
  echo ">>> (Optional) Cleaning conda/pip caches to reduce copy size..."
  "$OLD/bin/conda" clean --all -y >/dev/null 2>&1 || true
fi
find "$OLD/pkgs" -maxdepth 1 -name "*.tar.bz2" -delete 2>/dev/null || true
find "$OLD/pkgs" -maxdepth 1 -name "*.conda" -delete 2>/dev/null || true

echo ">>> Creating destination dir if needed..."
mkdir -p "$NEW"

# --- Copy (preserve perms, hardlinks, xattrs) ---
echo ">>> Copying $OLD -> $NEW ..."
rsync -aHAX --numeric-ids --info=progress2 "$OLD"/ "$NEW"/

# --- Verify (checksum dry-run; prints nothing if identical) ---
echo ">>> Verifying copy with rsync --checksum (dry-run)..."
RSYNC_DIFFS=$(rsync -aHAX --checksum --dry-run "$OLD"/ "$NEW"/ | wc -l | tr -d ' ')
if (( RSYNC_DIFFS > 0 )); then
  echo "ERROR: Verification failed; differences detected between $OLD and $NEW." >&2
  echo "       Aborting without deleting the old install." >&2
  exit 1
fi
echo ">>> Verification OK."

# --- Switch over: remove old to free root space ---
echo ">>> Removing old install to free / space..."
rm -rf "$OLD"

# --- Optional: compatibility symlink for hard-coded paths ---
if [ "${MAKE_COMPAT_SYMLINK}" = "yes" ]; then
  echo ">>> Creating compatibility symlink $OLD -> $NEW ..."
  ln -s "$NEW" "$OLD"
fi

# --- Update PATH lines in shell rc ---
echo ">>> Updating PATH in ${USER_SHELL_RC} ..."
touch "$USER_SHELL_RC"
# Remove any PATH entries pointing to OLD or NEW (avoid duplicates)
sed -i.bak "/${OLD//\//\\/}\/bin/d" "$USER_SHELL_RC"
sed -i "/${NEW//\//\\/}\/bin/d" "$USER_SHELL_RC"
# Add NEW to PATH
echo 'export PATH="'"$NEW"'/bin:$PATH"' >> "$USER_SHELL_RC"

# --- Make sure .condarc sends envs/pkgs to /data ---
CONDARC="${HOME}/.condarc"
echo ">>> Writing ${CONDARC} with envs/pkgs dirs ..."
cat > "$CONDARC" <<YAML
envs_dirs:
  - $NEW/envs
pkgs_dirs:
  - $NEW/pkgs
YAML

# --- Re-init conda (writes shell startup hooks) ---
echo ">>> Running conda init from NEW ..."
"$NEW/bin/conda" init bash >/dev/null 2>&1 || true
# For zsh users, also run:
# "$NEW/bin/conda" init zsh >/dev/null 2>&1 || true

# --- Final verify (in this shell) ---
echo ">>> Reloading shell rc and showing conda info ..."
# shellcheck disable=SC1090
source "$USER_SHELL_RC"
which conda || true
conda info || true

echo
echo "Done."
echo "Miniconda now lives at: $NEW"
if [ "${MAKE_COMPAT_SYMLINK}" = "yes" ]; then
  echo "A small symlink remains at: $OLD -> $NEW (safe; uses almost no space)."
fi
echo "Open a NEW shell and run: conda info"

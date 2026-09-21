#!/usr/bin/env bash
# Create ~/.config/prc2026/s3.env safely.
#
# Values are read from an interactive prompt rather than passed as arguments or
# typed into a heredoc, because anything typed at the shell prompt lands in
# ~/.bash_history. The secret is read with `read -s`, so it is never echoed to
# the terminal and never appears in history or in any process listing.
#
# In the MinIO console the two values come from Access Keys -> Create access key:
#   "Access Key"  -> PRC_S3_ACCESS_KEY
#   "Secret Key"  -> PRC_S3_SECRET_KEY   (shown only once, the password field)

set -euo pipefail

DIR="$HOME/.config/prc2026"
FILE="$DIR/s3.env"

mkdir -p "$DIR"
chmod 700 "$DIR"

if [[ -e "$FILE" ]]; then
    read -r -p "$FILE already exists. Overwrite? [y/N] " yn
    [[ "$yn" == [yY] ]] || { echo "unchanged."; exit 0; }
fi

read -r  -p "MinIO Access Key: " ACCESS
echo
read -rs -p "MinIO Secret Key (not echoed): " SECRET
echo

if [[ -z "$ACCESS" || -z "$SECRET" ]]; then
    echo "error: both values are required; nothing written." >&2
    exit 1
fi

# Create empty-and-private FIRST, then fill, so the secret is never briefly
# world-readable between creation and chmod.
install -m 600 /dev/null "$FILE"
printf 'PRC_S3_ACCESS_KEY=%s\nPRC_S3_SECRET_KEY=%s\n' "$ACCESS" "$SECRET" > "$FILE"
unset SECRET

echo
echo "wrote $FILE"
ls -l "$FILE"
echo
echo "Now verify with:  ./cluster/check_s3.sh"

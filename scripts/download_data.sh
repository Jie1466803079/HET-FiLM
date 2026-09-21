#!/usr/bin/env bash
# Fetches the HET-FiLM data payload from OneDrive UNSW into ${DATA_ROOT}.
#
# Reads scripts/data_manifest.txt — one file per line, format:
#   <relative_path>  <onedrive_download_url>  <sha256>  <size_bytes>
#
# Whitespace-separated; '#' comments and blank lines ignored.
# See DATA.md for the file manifest and OneDrive links.
#
# USAGE:
#   DATA_ROOT=/path/to/data bash scripts/download_data.sh
#
# Options:
#   HETFILM_SKIP_SHA256=1   skip the checksum verification step
#   HETFILM_ONLY=<glob>     only download paths matching this glob
#                           (e.g. HETFILM_ONLY='us/*' for US files only)

set -euo pipefail

: "${DATA_ROOT:?set DATA_ROOT to the directory where data should live}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${HETFILM_MANIFEST:-$SCRIPT_DIR/data_manifest.txt}"

if [[ ! -f "$MANIFEST" ]]; then
    echo "[download_data] ERROR: manifest not found at $MANIFEST" >&2
    exit 1
fi

check_sha() {
    local file="$1" expected="$2"
    [[ -n "$expected" && "$expected" != "<SHA256>" ]] || return 0
    local got
    got=$(sha256sum "$file" | awk '{print $1}')
    if [[ "$got" != "$expected" ]]; then
        echo "[download_data] SHA256 mismatch for $file: expected $expected, got $got" >&2
        return 1
    fi
    return 0
}

mkdir -p "$DATA_ROOT"

declare -i n_downloaded=0 n_skipped=0 n_failed=0

while read -r rel_path url expected_sha expected_size _rest; do
    [[ -z "$rel_path" || "$rel_path" == \#* ]] && continue
    if [[ -n "${HETFILM_ONLY:-}" && ! "$rel_path" == $HETFILM_ONLY ]]; then
        continue
    fi
    if [[ "$url" == "<ONEDRIVE_URL_HERE>" ]]; then
        echo "[download_data] SKIP $rel_path (manifest URL not filled in)"
        n_skipped+=1
        continue
    fi

    dest="$DATA_ROOT/$rel_path"
    mkdir -p "$(dirname "$dest")"

    # Skip if file present and correct size
    if [[ -f "$dest" ]]; then
        got_size=$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest")
        if [[ "$got_size" == "$expected_size" || "$expected_size" == "-" ]]; then
            if [[ -z "${HETFILM_SKIP_SHA256:-}" ]] && check_sha "$dest" "$expected_sha"; then
                echo "[download_data] SKIP $rel_path (already present, sha256 ok)"
                n_skipped+=1
                continue
            fi
        fi
    fi

    echo "[download_data] downloading $rel_path ($(numfmt --to=iec-i "$expected_size" 2>/dev/null || echo "$expected_size B")) → $dest"
    # -b "": OneDrive share links set a cookie on the first redirect that the
    # download request after it needs; this keeps it in memory for the transfer.
    if curl -L --fail --progress-bar -b "" --retry 5 --retry-delay 10 -o "$dest.tmp" "$url"; then
        mv "$dest.tmp" "$dest"
        if [[ -z "${HETFILM_SKIP_SHA256:-}" ]] && ! check_sha "$dest" "$expected_sha"; then
            n_failed+=1
            continue
        fi
        n_downloaded+=1
    else
        echo "[download_data] FAIL $rel_path" >&2
        rm -f "$dest.tmp"
        n_failed+=1
    fi
done < "$MANIFEST"

echo "[download_data] downloaded=$n_downloaded  skipped=$n_skipped  failed=$n_failed"
[[ $n_failed -eq 0 ]] || exit 1

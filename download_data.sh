#!/usr/bin/env bash
set -euo pipefail

URL="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
OUT_DIR="data"
OUT_FILE="$(basename "$URL")"

echo "Preparing to download: $URL"

# Ensure working directory exists
mkdir -p "$OUT_DIR"
cd "$OUT_DIR" || { echo "Cannot cd to $OUT_DIR"; exit 1; }

# Detect and use available downloader
download_file() {
	local url="$1" out="$2"
	if command -v aria2c >/dev/null 2>&1; then
		echo "Using aria2c to download"
		aria2c -x 16 -s 16 -k 1M -o "$out" "$url"
	elif command -v curl >/dev/null 2>&1; then
		echo "aria2c not found; using curl"
		curl -L --fail --output "$out" "$url"
	elif command -v wget >/dev/null 2>&1; then
		echo "aria2c/curl not found; using wget"
		wget -O "$out" "$url"
	else
		echo "No download utility found (aria2c / curl / wget). Please install one and retry." >&2
		return 1
	fi
}

if ! download_file "$URL" "$OUT_FILE"; then
	echo "Download failed; aborting." >&2
	exit 1
fi

if [ ! -s "$OUT_FILE" ]; then
	echo "Downloaded file $OUT_FILE is empty or missing; aborting." >&2
	exit 1
fi

echo "Download complete: $OUT_FILE"

if command -v tar >/dev/null 2>&1; then
	case "$OUT_FILE" in
		*.tar.bz2|*.tbz2)
			tar xjvf "$OUT_FILE" ;;
		*.tar.gz|*.tgz)
			tar xzvf "$OUT_FILE" ;;
		*.tar)
			tar xvf "$OUT_FILE" ;;
		*.zip)
			if command -v unzip >/dev/null 2>&1; then
				unzip -q "$OUT_FILE"
			else
				echo "unzip not found; cannot extract zip archive" >&2
				exit 1
			fi
			;;
		*)
			echo "Unknown archive format: $OUT_FILE" >&2
			exit 1
			;;
	esac
else
	echo "tar not found; cannot extract archives. Please install tar." >&2
	exit 1
fi

echo "Extraction complete. Cleaning up: $OUT_FILE"
rm -f "$OUT_FILE"

echo "Done. Data is available in $(pwd)"

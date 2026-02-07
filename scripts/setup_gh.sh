#!/bin/bash
set -euo pipefail

# Setup script to install GitHub CLI (gh) for Claude Code sessions.
# This script is intended to be run automatically via .claude/hooks.json
# on session start.

GH_VERSION="${GH_VERSION:-2.67.0}"

if command -v gh &>/dev/null; then
    echo "gh is already installed: $(gh --version | head -1)"
    exit 0
fi

echo "Installing GitHub CLI (gh) v${GH_VERSION}..."

# Detect OS and architecture
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

if [[ "$OS" != "linux" ]]; then
    echo "Error: This script only supports Linux." >&2
    exit 1
fi

case "$ARCH" in
    x86_64)  ARCH="amd64" ;;
    aarch64) ARCH="arm64" ;;
    *)
        echo "Error: Unsupported architecture: $ARCH" >&2
        exit 1
        ;;
esac

TARBALL="gh_${GH_VERSION}_${OS}_${ARCH}.tar.gz"
URL="https://github.com/cli/cli/releases/download/v${GH_VERSION}/${TARBALL}"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "Downloading ${URL}..."
curl -fsSL "$URL" -o "${TMPDIR}/${TARBALL}"

tar -xzf "${TMPDIR}/${TARBALL}" -C "$TMPDIR"

# Install binary to /usr/local/bin
cp "${TMPDIR}/gh_${GH_VERSION}_${OS}_${ARCH}/bin/gh" /usr/local/bin/gh
chmod +x /usr/local/bin/gh

echo "gh installed successfully: $(gh --version | head -1)"

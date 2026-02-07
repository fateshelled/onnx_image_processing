#!/bin/bash
set -euo pipefail

# Setup script to install GitHub CLI (gh) for Claude Code sessions.
# This script is intended to be run automatically via .claude/hooks.json
# on session start.

GH_VERSION="${GH_VERSION:-2.67.0}"

# Validate version format to prevent path traversal or injection
if [[ ! "$GH_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Invalid GH_VERSION format: $GH_VERSION (expected: X.Y.Z)" >&2
    exit 1
fi

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
RELEASE_BASE="https://github.com/cli/cli/releases/download/v${GH_VERSION}"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "Downloading ${RELEASE_BASE}/${TARBALL}..."
curl -fsSL "${RELEASE_BASE}/${TARBALL}" -o "${TMPDIR}/${TARBALL}"
curl -fsSL "${RELEASE_BASE}/gh_${GH_VERSION}_checksums.txt" -o "${TMPDIR}/checksums.txt"

# Verify checksum
EXPECTED_SHA="$(grep "${TARBALL}" "${TMPDIR}/checksums.txt" | awk '{print $1}')"
if [[ -z "$EXPECTED_SHA" ]]; then
    echo "Error: Checksum not found for ${TARBALL}" >&2
    exit 1
fi
ACTUAL_SHA="$(sha256sum "${TMPDIR}/${TARBALL}" | awk '{print $1}')"
if [[ "$EXPECTED_SHA" != "$ACTUAL_SHA" ]]; then
    echo "Error: Checksum mismatch!" >&2
    echo "  Expected: $EXPECTED_SHA" >&2
    echo "  Actual:   $ACTUAL_SHA" >&2
    exit 1
fi
echo "Checksum verified."

# Extract and install binary
EXTRACT_DIR="${TMPDIR}/gh"
mkdir -p "$EXTRACT_DIR"
tar -xzf "${TMPDIR}/${TARBALL}" -C "$EXTRACT_DIR" --strip-components=1

install -m 755 "${EXTRACT_DIR}/bin/gh" /usr/local/bin/gh

echo "gh installed successfully: $(gh --version | head -1)"

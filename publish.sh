#!/usr/bin/env bash
# Build and publish curry-leaves to PyPI.
#
# Usage:
#   ./publish.sh                              # publish current version in pyproject.toml
#   ./publish.sh patch|minor|major|<version>  # bump version, commit, tag, then publish
#   ./publish.sh --dry-run                    # run everything except the actual upload
#
# Auth: `twine upload` uses ~/.pypirc or TWINE_USERNAME/TWINE_PASSWORD
# (for a PyPI API token: TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-...).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PY="${PYTHON:-python3}"
if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python; fi

DRY_RUN=false
BUMP=""

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    *) BUMP="$arg" ;;
  esac
done

echo "==> Checking git working tree"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: working tree is not clean. Commit or stash changes before publishing." >&2
  git status --short
  exit 1
fi

current_version() {
  "$PY" -c 'import tomllib; print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])'
}

if [[ -n "$BUMP" ]]; then
  OLD_VERSION="$(current_version)"
  NEW_VERSION="$("$PY" - "$OLD_VERSION" "$BUMP" <<'EOF'
import re, sys
old, bump = sys.argv[1], sys.argv[2]
if re.fullmatch(r"\d+\.\d+\.\d+", bump):
    print(bump)
else:
    major, minor, patch = map(int, old.split("."))
    if bump == "major":   major, minor, patch = major + 1, 0, 0
    elif bump == "minor": minor, patch = minor + 1, 0
    elif bump == "patch": patch += 1
    else: sys.exit(f"error: unknown bump '{bump}' (use patch|minor|major|X.Y.Z)")
    print(f"{major}.{minor}.{patch}")
EOF
)"
  echo "==> Bumping version $OLD_VERSION -> $NEW_VERSION"
  "$PY" - "$OLD_VERSION" "$NEW_VERSION" <<'EOF'
import sys
old, new = sys.argv[1], sys.argv[2]
for path, pattern in [
    ("pyproject.toml", 'version = "{}"'),
    ("src/curry_leaves/__init__.py", 'VERSION = "{}"'),
]:
    text = open(path).read()
    needle, repl = pattern.format(old), pattern.format(new)
    if needle not in text:
        sys.exit(f"error: {needle!r} not found in {path}")
    open(path, "w").write(text.replace(needle, repl, 1))
EOF
  git add pyproject.toml src/curry_leaves/__init__.py
  git commit -m "chore: release v$NEW_VERSION"
  git tag "v$NEW_VERSION"
fi

echo "==> Installing dev dependencies"
"$PY" -m pip install -q -e ".[dev]"

echo "==> Type-checking"
"$PY" -m mypy src

echo "==> Running tests"
"$PY" -m pytest -q

echo "==> Building sdist + wheel"
rm -rf dist build
"$PY" -m build

echo "==> Checking distribution metadata"
"$PY" -m twine check dist/*

PKG_VERSION="$(current_version)"

if $DRY_RUN; then
  echo "==> Dry run: skipping twine upload"
  exit 0
fi

echo "==> Publishing curry-leaves@$PKG_VERSION to PyPI"
"$PY" -m twine upload dist/*

if [[ -n "$BUMP" ]]; then
  echo "==> Pushing commit and tag"
  git push
  git push --tags
fi

echo "==> Done: published curry-leaves@$PKG_VERSION"

#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Usage as a pre-push hook:
#   ln -sf ../../update_readme_structure.sh .git/hooks/pre-push
# Or run manually before pushing:
#   ./update_readme_structure.sh && git push origin main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
for cmd in git tree awk; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "❌ Required command not found: $cmd"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Ensure README.md exists in current directory
# ---------------------------------------------------------------------------
if [ ! -f "README.md" ]; then
    echo "❌ README.md not found in current directory ($(pwd))."
    echo "   Run this script from the project root."
    exit 1
fi

# ---------------------------------------------------------------------------
# Temporary files with guaranteed cleanup
# ---------------------------------------------------------------------------
TMP_TREE=$(mktemp)
TMP_README=$(mktemp)
trap 'rm -f "$TMP_TREE" "$TMP_README"' EXIT

# ---------------------------------------------------------------------------
# Generate directory tree
# Excludes: hidden dirs (.claude, .github), logs, venv, binaries, caches
# Strips the summary line ("N directories, M files") added by tree
# ---------------------------------------------------------------------------
git ls-files | tree --fromfile -F -a -L 4 --dirsfirst \
    -I "logs|venv|*.csv|*.pkl|*.png|__pycache__|*.code-workspace|*.ipynb|.vscode|.gitkeep|.DS_Store|.github|.claude" \
    | grep -Ev "^[0-9]+ director" \
    > "$TMP_TREE"

if [ ! -s "$TMP_TREE" ]; then
    echo "❌ tree produced no output — README not modified."
    exit 1
fi

# ---------------------------------------------------------------------------
# Update README between markers
# ---------------------------------------------------------------------------
awk -v tmp="$TMP_TREE" '
/<!-- PROJECT_STRUCTURE_START -->/ {
    print
    print "```text"
    while ((getline line < tmp) > 0) print line
    close(tmp)
    print "```"
    skip = 1
    next
}
/<!-- PROJECT_STRUCTURE_END -->/ { skip = 0 }
!skip
' README.md > "$TMP_README"

mv "$TMP_README" README.md

# ---------------------------------------------------------------------------
# Auto-commit README if it changed (useful when run as pre-push hook)
# ---------------------------------------------------------------------------
if ! git diff --quiet README.md; then
    git add README.md
    git commit -m "Auto-update project structure in README"
    echo "✅ README updated and committed."
else
    echo "✅ README already up to date."
fi

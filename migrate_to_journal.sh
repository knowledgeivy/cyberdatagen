#!/bin/bash
# migrate_to_journal.sh
# Copies relevant files from cyberdata (conference) to cyberdata-journal (new repo).
# Run this from the root of the cyberdata repo.
# Usage: bash migrate_to_journal.sh

set -e

SRC="$(cd "$(dirname "$0")" && pwd)"
DST="$HOME/Notebooks/cyberdata-journal"

echo "========================================"
echo "  cyberdata → cyberdata-journal"
echo "  SRC: $SRC"
echo "  DST: $DST"
echo "========================================"

# Abort if target already exists and is non-empty
if [ -d "$DST" ] && [ "$(ls -A "$DST" 2>/dev/null)" ]; then
    echo "ERROR: $DST already exists and is non-empty."
    echo "Delete it manually or choose a different path."
    exit 1
fi

mkdir -p "$DST"

# ── Core source code ──────────────────────────────────────────────
echo "[1/7] Copying src/ ..."
cp -r "$SRC/src" "$DST/src"

# ── Pipeline scripts ──────────────────────────────────────────────
echo "[2/7] Copying scripts/ ..."
cp -r "$SRC/scripts" "$DST/scripts"

# ── Config files → templates only ────────────────────────────────
echo "[3/7] Copying config/ as templates ..."
mkdir -p "$DST/config/templates"
cp "$SRC"/config/*.yaml "$DST/config/templates/" 2>/dev/null || true

# ── Conference paper → read-only archive ─────────────────────────
echo "[4/7] Copying paper/ → paper_conference/ ..."
cp -r "$SRC/paper" "$DST/paper_conference"

# ── Dotfiles ──────────────────────────────────────────────────────
echo "[5/7] Copying dotfiles ..."
[ -f "$SRC/.gitignore" ]  && cp "$SRC/.gitignore"  "$DST/.gitignore"
[ -f "$SRC/.env.public" ] && cp "$SRC/.env.public" "$DST/.env.public"

# ── EXTENSION_PLAN (for reference) ───────────────────────────────
[ -f "$SRC/EXTENSION_PLAN.md" ] && cp "$SRC/EXTENSION_PLAN.md" "$DST/EXTENSION_PLAN.md"

# ── Create empty directory structure ─────────────────────────────
echo "[6/7] Creating directory structure ..."
mkdir -p "$DST/data/raw/enron"
mkdir -p "$DST/output"
mkdir -p "$DST/logs"
mkdir -p "$DST/paper/sections"
mkdir -p "$DST/paper/pic"
mkdir -p "$DST/config"

# Keep empty dirs in git
touch "$DST/data/raw/enron/.gitkeep"
touch "$DST/output/.gitkeep"
touch "$DST/logs/.gitkeep"

# ── Copy pre-generated CLAUDE.md and pyproject.toml ──────────────
echo "[7/7] Copying CLAUDE.md and pyproject.toml ..."
[ -f "$SRC/CLAUDE.md" ]               && cp "$SRC/CLAUDE.md"               "$DST/CLAUDE.md"
[ -f "$SRC/pyproject_journal.toml" ]  && cp "$SRC/pyproject_journal.toml"  "$DST/pyproject.toml"

echo ""
echo "========================================"
echo "  Migration complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Create GitHub repo 'cyberdata-journal' (private), then:"
echo ""
echo "     cd $DST"
echo "     git init"
echo "     git remote add origin git@github.com:<username>/cyberdata-journal.git"
echo ""
echo "  2. Set up uv environment:"
echo ""
echo "     cd $DST"
echo "     curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv not installed"
echo "     uv python pin 3.12"
echo "     uv sync"
echo ""
echo "  3. Create .env with your API keys:"
echo ""
echo "     cp $DST/.env.public $DST/.env"
echo "     # Then edit .env and fill in:"
echo "     #   OPENAI_API_KEY=sk-..."
echo "     #   ANTHROPIC_API_KEY=sk-ant-..."
echo "     #   OPENROUTER_API_KEY=sk-or-..."
echo ""
echo "  4. Download Enron dataset:"
echo ""
echo "     cd $DST"
echo "     uv run python -c \""
echo "     from datasets import load_dataset"
echo "     ds = load_dataset('SetFit/enron_spam')"
echo "     ds.save_to_disk('data/raw/enron')"
echo "     print('Enron dataset ready:', ds)"
echo "     \""
echo ""
echo "  5. First commit:"
echo ""
echo "     cd $DST"
echo "     git add ."
echo "     git commit -m 'Initial commit: journal extension of ICISSP 2026 paper'"
echo "     git push -u origin main"
echo ""

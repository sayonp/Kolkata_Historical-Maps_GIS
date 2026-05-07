#!/usr/bin/env bash
# PhD Research Environment Setup & Session Launcher
# Run once to scaffold, then daily as session launcher
# Author: Spandita Pramanik | UCL Bartlett
# Usage: bash phd_setup.sh [init|start|end]

set -e

# ─── CONFIG ────────────────────────────────────────────────────────────────────
VAULT="$HOME/Documents/phd-vault"
REPO="$HOME/Documents/colonial-map-pipeline"
NOTION_URL="https://notion.so/35682b180df181c2b9cfdcba5718ab76"
UCL_OD="/Users/xon/UCL-OD"

# ─── COLOURS ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: init — run once to scaffold entire environment
# ═══════════════════════════════════════════════════════════════════════════════
init_environment() {
    info "Scaffolding PhD research environment..."

    # 1. Obsidian vault directories
    mkdir -p "$VAULT"/{handoffs,theory,methodology,data,literature,code,admin}
    success "Obsidian vault structure created at $VAULT"

    # 2. Git repo directories
    mkdir -p "$REPO"/{data/raw/{maps,newspapers},data/processed/{georeferenced,routes},data/syntax,scripts,notebooks/exploratory,tests,outputs/{figures,reports},docs}
    success "Repo directory structure created at $REPO"

    # 3. Move existing assets if present
    if [ -d "$UCL_OD/Maps" ]; then
        info "Symlinking UCL OneDrive maps..."
        ln -sfn "$UCL_OD/Maps" "$REPO/data/raw/maps/ucl_od_maps"
        success "UCL OD maps linked"
    else
        warn "UCL OneDrive Maps not found at $UCL_OD/Maps — link manually"
    fi

    # 4. Init git if not already
    if [ ! -d "$REPO/.git" ]; then
        cd "$REPO"
        git init
        git checkout -b main
        success "Git repo initialized"
    else
        info "Git repo already initialized"
    fi

    # 5. Create tests.json tracker
    cat > "$REPO/tests/tests.json" << 'EOF'
{
  "scripts": [
    {"script": "01_map_ingestion.py",    "test_file": "tests/test_01.py", "status": "untested", "last_run": null, "coverage": [], "blocker": null},
    {"script": "02_georeferencing.py",   "test_file": "tests/test_02.py", "status": "untested", "last_run": null, "coverage": [], "blocker": null},
    {"script": "03_nlp_extraction.py",   "test_file": "tests/test_03.py", "status": "untested", "last_run": null, "coverage": [], "blocker": null},
    {"script": "04_route_mapping.py",    "test_file": "tests/test_04.py", "status": "untested", "last_run": null, "coverage": [], "blocker": null},
    {"script": "05_syntax_analysis.py",  "test_file": "tests/test_05.py", "status": "untested", "last_run": null, "coverage": [], "blocker": null},
    {"script": "06_abm_simulation.py",   "test_file": "tests/test_06.py", "status": "untested", "last_run": null, "coverage": [], "blocker": null}
  ]
}
EOF
    success "tests.json initialized"

    # 6. Create README with links
    cat > "$REPO/README.md" << EOF
# Colonial Map Pipeline — Kolkata Processional Routes

**PI:** Spandita Pramanik | UCL Bartlett Space Syntax Lab

## Quick Links
- **Notion Hub:** $NOTION_URL
- **Obsidian Vault:** $VAULT
- **Latest Handoff:** $VAULT/handoffs/LATEST_HANDOFF.md
- **Master Process Flow:** docs/MASTER_PROCESS_FLOW.md

## Session Start
\`\`\`bash
bash phd_setup.sh start
\`\`\`

## Session End
\`\`\`bash
bash phd_setup.sh end
\`\`\`
EOF
    success "README.md created"

    # 7. Create Obsidian INDEX
    cat > "$VAULT/00_INDEX.md" << EOF
# PhD Research Index
**Repo:** $REPO
**Notion:** $NOTION_URL
**Last updated:** $(date +%Y-%m-%d)

## Quick Navigation
- [[handoffs/LATEST_HANDOFF]] — Current session state
- [[methodology/pipeline_architecture]] — Full pipeline diagram
- [[literature/reading_log]] — Paper tracker
- [[code/script_log]] — Script status
- [[data/procession_routes]] — Route inventory
- [[admin/ucl_website_profile]] — Research profile

## Pipeline Status
| Stage | Script | Status |
|-------|--------|--------|
| Map Acquisition | 01_map_ingestion.py | 🔴 Not started |
| Georeferencing | 02_georeferencing.py | 🔴 Not started |
| NLP Extraction | 03_nlp_extraction.py | 🔴 Not started |
| Route Mapping | 04_route_mapping.py | 🔴 Not started |
| Syntax Analysis | 05_syntax_analysis.py | 🔴 Not started |
| ABM Simulation | 06_abm_simulation.py | 🔴 Not started |
EOF
    success "Obsidian INDEX created"

    # 8. Create empty log files
    touch "$VAULT/literature/reading_log.md"
    touch "$VAULT/literature/gaps.md"
    touch "$VAULT/code/script_log.md"
    touch "$VAULT/data/procession_routes.md"
    touch "$VAULT/data/map_inventory.md"
    touch "$VAULT/admin/ucl_website_profile.md"

    # 9. Initial git commit
    cd "$REPO"
    git add .
    git commit -m "[E][$(date +%Y%m%d)] INIT: Scaffold repo and vault structure

Obsidian-ref: VAULT_INIT
Notion-page: 35682b18
Next: Begin Script 01 — map ingestion pipeline
Blockers: none"

    success "Initial commit made"

    echo ""
    echo -e "${GREEN}════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Environment initialized successfully!  ${NC}"
    echo -e "${GREEN}════════════════════════════════════════${NC}"
    echo "  Next step: Run 'bash phd_setup.sh start' to begin a session"
}

# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: start — cold-start protocol for every session
# ═══════════════════════════════════════════════════════════════════════════════
start_session() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}   PhD SESSION COLD-START                          ${NC}"
    echo -e "${BLUE}   $(date '+%A %d %B %Y, %H:%M')                  ${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
    echo ""

    # ── Git State ──
    info "GIT STATE (last 10 commits):"
    cd "$REPO"
    git log --oneline -10
    echo ""
    info "GIT STATUS:"
    git status --short
    echo ""

    # ── Latest Handoff ──
    HANDOFF_LINK="$VAULT/handoffs/LATEST_HANDOFF.md"
    if [ -f "$HANDOFF_LINK" ]; then
        info "LATEST HANDOFF:"
        echo -e "${YELLOW}$(cat "$HANDOFF_LINK")${NC}"
    else
        warn "No handoff file found. This may be the first session."
        warn "Check Notion: $NOTION_URL"
    fi
    echo ""

    # ── Session Type ──
    echo -e "${BLUE}Select session type:${NC}"
    echo "  A) Literature    B) Code/Data    C) Analysis"
    echo "  D) Writing       E) Planning"
    read -rp "  → Enter letter: " SESSION_TYPE
    SESSION_TYPE="${SESSION_TYPE^^}"  # uppercase

    # Validate
    case "$SESSION_TYPE" in
        A|B|C|D|E) success "Session type: $SESSION_TYPE" ;;
        *) error "Invalid session type. Use A, B, C, D, or E." ;;
    esac

    # Store for end-of-session use
    echo "$SESSION_TYPE" > /tmp/phd_session_type

    # ── Branch recommendation ──
    echo ""
    info "Recommended git branch for this session type:"
    case "$SESSION_TYPE" in
        A) echo "  → git checkout dev  (literature notes don't need feature branches)" ;;
        B) echo "  → git checkout -b feat/<slug>  (e.g. feat/nlp-extraction-v2)" ;;
        C) echo "  → git checkout dev" ;;
        D) echo "  → git checkout dev  (writing drafts)" ;;
        E) echo "  → git checkout main  (planning review)" ;;
    esac

    # ── Open tools ──
    echo ""
    info "Opening Notion hub..."
    open "$NOTION_URL" 2>/dev/null || warn "Could not auto-open Notion (run on Mac)"

    echo ""
    echo -e "${GREEN}Session ready. Go.${NC}"
}

# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: end — end-of-session handoff protocol
# ═══════════════════════════════════════════════════════════════════════════════
end_session() {
    echo ""
    echo -e "${YELLOW}═══════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}   SESSION END — HANDOFF PROTOCOL                  ${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════${NC}"
    echo ""

    # Retrieve session type
    SESSION_TYPE=$(cat /tmp/phd_session_type 2>/dev/null || echo "B")
    DATE=$(date +%Y%m%d)

    read -rp "Handoff slug (e.g. nlp-regex-debug): " SLUG
    HANDOFF_FILE="$VAULT/handoffs/${DATE}_handoff_${SLUG}.md"

    # Get current git commit
    cd "$REPO"
    GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "no-commits")
    BRANCH=$(git branch --show-current)

    # Write handoff template
    cat > "$HANDOFF_FILE" << EOF
---
date: $(date +%Y-%m-%d)
session_type: $SESSION_TYPE
git_commit: $GIT_SHA
git_branch: $BRANCH
notion_page_id: 35682b18
status: active
next_action: "FILL THIS IN BEFORE CLOSING"
---

## What Was Done
- 

## State of Each Component
- Script 01 (map ingestion): 
- Script 02 (georeferencing): 
- Script 03 (NLP extraction): 
- Script 04 (route mapping): 
- Script 05 (syntax analysis): 
- Script 06 (ABM simulation): 
- Obsidian vault: 
- Notion board: 

## Blockers
- 

## Next Session Must Start With
1. 
2. 
3. 

## Do Not Forget
- 
EOF

    # Open in default editor
    ${EDITOR:-nano} "$HANDOFF_FILE"

    # Update symlink
    ln -sf "$HANDOFF_FILE" "$VAULT/handoffs/LATEST_HANDOFF.md"
    success "Handoff file written and symlinked as LATEST_HANDOFF.md"

    # Prompt for commit
    echo ""
    read -rp "Commit message description: " COMMIT_DESC
    read -rp "Next step (one line): " NEXT_STEP
    read -rp "Blockers (or 'none'): " BLOCKERS

    git add .
    git commit -m "[$SESSION_TYPE][$DATE] $COMMIT_DESC

Obsidian-ref: ${DATE}_handoff_${SLUG}.md
Notion-page: 35682b18
Next: $NEXT_STEP
Blockers: $BLOCKERS"

    read -rp "Push to remote? (y/n): " PUSH
    if [[ "$PUSH" == "y" ]]; then
        git push origin "$BRANCH"
        success "Pushed to $BRANCH"
    fi

    echo ""
    echo -e "${GREEN}Session closed cleanly. Memory persisted.${NC}"
    echo -e "  Handoff: ${YELLOW}$HANDOFF_FILE${NC}"
    echo -e "  Git SHA: ${YELLOW}$GIT_SHA${NC}"
}

# ─── DISPATCH ──────────────────────────────────────────────────────────────────
case "${1:-}" in
    init)  init_environment ;;
    start) start_session ;;
    end)   end_session ;;
    *)
        echo "Usage: bash phd_setup.sh [init|start|end]"
        echo "  init  — scaffold vault + repo structure (run once)"
        echo "  start — cold-start protocol (run at beginning of every session)"
        echo "  end   — handoff protocol (run at end of every session)"
        ;;
esac

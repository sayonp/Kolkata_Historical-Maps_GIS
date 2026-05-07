# Complete Setup Guide — PhD Research Environment
**Spandita Pramanik | UCL Bartlett Space Syntax Lab**  
**Time required: ~45–60 minutes (mostly waiting for installs)**

---

## WHAT YOU'RE BUILDING

By the end of this guide you will have:

1. `colonial-map-pipeline` → GitHub (private) — code, scripts, data
2. `phd-vault` → GitHub (private) — Obsidian notes, handoffs, theory
3. `phd-vault` → Obsidian Sync (cloud) — live sync across Mac + any other device
4. `phd_setup.sh` → wired to both, so `start` and `end` commands handle everything

The vault has **two independent backups**: if GitHub goes down, Obsidian Sync works. If Obsidian Sync lapses, Git works. Neither depends on the other.

---

## PREREQUISITES CHECK

Open Terminal and run each line. Fix anything that returns "not found" before proceeding.

```bash
# Check Git
git --version
# Expected: git version 2.x.x — if missing: brew install git

# Check GitHub CLI (needed for repo creation without browser)
gh --version
# If missing:
brew install gh

# Check if you have a GitHub account
# → Go to github.com and confirm you're logged in

# Check Homebrew (Mac package manager)
brew --version
# If missing: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Check Python (for later pipeline work)
python3 --version
# Expected: Python 3.10+ — if missing: brew install python
```

---

## PHASE 1 — SCAFFOLD YOUR LOCAL DIRECTORIES

### Step 1.1 — Create the vault directory

```bash
mkdir -p ~/Documents/phd-vault/{handoffs,theory,methodology,data,literature,code,admin}
```

Verify:
```bash
ls ~/Documents/phd-vault/
# Should show: admin  code  data  handoffs  literature  methodology  theory
```

### Step 1.2 — Create the pipeline repo directory

```bash
mkdir -p ~/Documents/colonial-map-pipeline/{data/raw/{maps,newspapers},data/processed/{georeferenced,routes},data/syntax,scripts,notebooks/exploratory,tests,outputs/{figures,reports},docs}
```

Verify:
```bash
ls ~/Documents/colonial-map-pipeline/
# Should show: data  docs  notebooks  outputs  scripts  tests
```

### Step 1.3 — Create core files in both

```bash
# Vault index
cat > ~/Documents/phd-vault/00_INDEX.md << 'EOF'
# PhD Research Index
**Repo:** ~/Documents/colonial-map-pipeline
**Notion:** https://notion.so/35682b180df181c2b9cfdcba5718ab76
**Last updated:** 2026-05-07

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

# Touch all log files so they exist
touch ~/Documents/phd-vault/literature/reading_log.md
touch ~/Documents/phd-vault/literature/gaps.md
touch ~/Documents/phd-vault/code/script_log.md
touch ~/Documents/phd-vault/data/procession_routes.md
touch ~/Documents/phd-vault/data/map_inventory.md
touch ~/Documents/phd-vault/admin/ucl_website_profile.md
touch ~/Documents/phd-vault/methodology/pipeline_architecture.md

# Repo README
cat > ~/Documents/colonial-map-pipeline/README.md << 'EOF'
# Colonial Map Pipeline — Kolkata Processional Routes

**PI:** Spandita Pramanik | UCL Bartlett Space Syntax Lab

## Quick Links
- **Notion Hub:** https://notion.so/35682b180df181c2b9cfdcba5718ab76
- **Obsidian Vault:** ~/Documents/phd-vault
- **Latest Handoff:** ~/Documents/phd-vault/handoffs/LATEST_HANDOFF.md
- **Master Process Flow:** docs/MASTER_PROCESS_FLOW.md

## Session Commands
```bash
bash phd_setup.sh start   # Cold-start every session
bash phd_setup.sh end     # Handoff + commit at end
```
EOF

# tests.json tracker
cat > ~/Documents/colonial-map-pipeline/tests/tests.json << 'EOF'
{
  "scripts": [
    {"script": "01_map_ingestion.py",   "test_file": "tests/test_01.py", "status": "untested", "last_run": null, "coverage": [], "blocker": null},
    {"script": "02_georeferencing.py",  "test_file": "tests/test_02.py", "status": "untested", "last_run": null, "coverage": [], "blocker": null},
    {"script": "03_nlp_extraction.py",  "test_file": "tests/test_03.py", "status": "untested", "last_run": null, "coverage": [], "blocker": null},
    {"script": "04_route_mapping.py",   "test_file": "tests/test_04.py", "status": "untested", "last_run": null, "coverage": [], "blocker": null},
    {"script": "05_syntax_analysis.py", "test_file": "tests/test_05.py", "status": "untested", "last_run": null, "coverage": [], "blocker": null},
    {"script": "06_abm_simulation.py",  "test_file": "tests/test_06.py", "status": "untested", "last_run": null, "coverage": [], "blocker": null}
  ]
}
EOF

echo "✅ All files created"
```

---

## PHASE 2 — GIT: INITIALIZE BOTH LOCAL REPOS

### Step 2.1 — Init the pipeline repo

```bash
cd ~/Documents/colonial-map-pipeline
git init
git checkout -b main

# Create .gitignore to exclude large raw data from git
# (data lives on UCL OneDrive + backed up separately)
cat > .gitignore << 'EOF'
# Raw data — too large for git, backed up to UCL OneDrive
data/raw/maps/*.tif
data/raw/maps/*.tiff
data/raw/newspapers/*.pdf

# Python
__pycache__/
*.pyc
.env
venv/
.venv/

# QGIS
*.qgz.bak
*.qgd

# OS
.DS_Store
EOF

git add .
git commit -m "[E][20260507] INIT: Scaffold colonial-map-pipeline repo

Obsidian-ref: VAULT_INIT
Notion-page: 35682b18
Next: Create GitHub remote and push
Blockers: none"

echo "✅ Pipeline repo initialized"
```

### Step 2.2 — Init the vault repo

```bash
cd ~/Documents/phd-vault
git init
git checkout -b main

# .gitignore for vault — exclude any accidental sensitive files
cat > .gitignore << 'EOF'
# Obsidian app config (not notes)
.obsidian/workspace.json
.obsidian/workspace-mobile.json

# OS
.DS_Store
*.tmp
EOF

git add .
git commit -m "[E][20260507] INIT: Scaffold phd-vault Obsidian repository

Next: Create GitHub remote and push
Blockers: none"

echo "✅ Vault repo initialized"
```

---

## PHASE 3 — GITHUB: CREATE PRIVATE REMOTE REPOS

### Step 3.1 — Authenticate GitHub CLI

```bash
gh auth login
```

You will be asked:
- **What account?** → GitHub.com
- **Preferred protocol?** → HTTPS
- **Authenticate with browser?** → Yes

A browser window opens. Log in to GitHub. Come back to Terminal when it says "Logged in."

Verify:
```bash
gh auth status
# Should show: ✓ Logged in to github.com
```

### Step 3.2 — Create the pipeline repo on GitHub

```bash
cd ~/Documents/colonial-map-pipeline

gh repo create colonial-map-pipeline \
  --private \
  --description "Kolkata processional routes: georeferencing, NLP extraction, space syntax analysis — UCL PhD" \
  --source=. \
  --remote=origin \
  --push

echo "✅ Pipeline repo pushed to GitHub"
```

Verify it worked:
```bash
git remote -v
# Should show: origin  https://github.com/YOUR_USERNAME/colonial-map-pipeline.git
```

### Step 3.3 — Create the vault repo on GitHub

```bash
cd ~/Documents/phd-vault

gh repo create phd-vault \
  --private \
  --description "Obsidian vault: PhD research notes, handoffs, theory — UCL Bartlett Space Syntax Lab" \
  --source=. \
  --remote=origin \
  --push

echo "✅ Vault repo pushed to GitHub"
```

Verify:
```bash
git remote -v
# Should show: origin  https://github.com/YOUR_USERNAME/phd-vault.git
```

---

## PHASE 4 — OBSIDIAN: INSTALL AND CONFIGURE

### Step 4.1 — Install Obsidian

Go to: **https://obsidian.md/download**

Download the macOS version (.dmg). Open it, drag to Applications. Launch Obsidian.

### Step 4.2 — Open your vault in Obsidian

When Obsidian opens:
1. Click **"Open folder as vault"**
2. Navigate to: `~/Documents/phd-vault`
3. Click **Open**

You should see your notes in the left sidebar: `00_INDEX`, `handoffs/`, `theory/`, etc.

### Step 4.3 — Enable core plugins you need

In Obsidian: `Settings (⌘,)` → **Core plugins** → Turn ON:
- ✅ Templates
- ✅ Daily notes (optional but useful)
- ✅ Graph view (visualise note links)
- ✅ Backlinks
- ✅ Outgoing links

### Step 4.4 — Install community plugins

In Obsidian: `Settings` → **Community plugins** → Turn off Restricted mode → Browse

Install these:
1. **Git** (by Vinzent Steinberg) — auto-commits vault to GitHub on schedule
2. **Dataview** — query your notes like a database (useful for tracking script status)
3. **Templater** — powerful templates for handoff files

After installing each: click **Enable**.

### Step 4.5 — Configure the Git plugin

`Settings` → **Community plugins** → Git → Settings icon

Set:
```
Vault backup interval (minutes): 30
Auto pull interval (minutes): 0  (pull manually at session start)
Commit message: "vault: auto-backup {{date}}"
Pull updates on startup: ✅ ON
Push on backup: ✅ ON
```

This means: every 30 minutes while Obsidian is open, your vault commits and pushes to GitHub automatically. You never lose more than 30 minutes of notes.

### Step 4.6 — Create the handoff template

In Obsidian, create a new file at: `Templates/handoff_template.md`

Paste this content:
```markdown
---
date: {{date:YYYY-MM-DD}}
session_type: 
git_commit: 
git_branch: 
notion_page_id: 35682b18
status: active
next_action: "FILL THIS IN"
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

## Blockers
- 

## Next Session Must Start With
1. 
2. 
3. 

## Do Not Forget
- 
```

Then in `Settings` → **Templates** → set Template folder to `Templates`.

Now when you need a handoff: `⌘P` → "Insert template" → `handoff_template`.

---

## PHASE 5 — OBSIDIAN SYNC: SET UP CLOUD BACKUP

Obsidian Sync is **separate from the Git plugin**. Git backs up to GitHub (version controlled). Sync backs up to Obsidian's servers (real-time, works on mobile).

### Step 5.1 — Purchase Obsidian Sync

Go to: **https://obsidian.md/sync**

Cost: $8/month or $96/year (worth it — this is your primary research memory).

Create an Obsidian account at **https://obsidian.md/account** if you don't have one.

### Step 5.2 — Enable Sync in Obsidian

In Obsidian: `Settings` → **Core plugins** → **Sync** → Enable

Then: `Settings` → **Sync** (appears in left panel after enabling)

1. Click **Log in** → enter your Obsidian account credentials
2. Click **Create new vault** (in the Sync panel, not the vault panel)
3. Name it: `phd-vault`
4. Set encryption passphrase: **choose something strong, write it down somewhere safe — Obsidian cannot recover this if lost**
5. Click **Create**

### Step 5.3 — Configure what gets synced

In `Settings` → **Sync**:

```
Sync settings:          ✅ ON
Sync appearance:        ✅ ON  
Sync themes:            ✅ ON
Sync snippets:          ✅ ON
Excluded folders:       (leave blank)
```

### Step 5.4 — Verify sync is working

Look at the bottom-right of Obsidian. You should see a green cloud icon or "Sync: Up to date."

Make a small change to `00_INDEX.md` (add a space, save). Watch the cloud icon. Within 10 seconds it should show "Syncing…" then "Up to date."

✅ Sync is live.

---

## PHASE 6 — WIRE phd_setup.sh TO BOTH REPOS

### Step 6.1 — Save the setup script

```bash
# Copy the setup script to a permanent location
cp ~/Downloads/phd_setup.sh ~/Documents/phd_setup.sh
chmod +x ~/Documents/phd_setup.sh

# Also copy the master process flow doc into both repos
cp ~/Downloads/PhD_MASTER_PROCESS_FLOW.md \
   ~/Documents/colonial-map-pipeline/docs/MASTER_PROCESS_FLOW.md

cp ~/Downloads/PhD_MASTER_PROCESS_FLOW.md \
   ~/Documents/phd-vault/methodology/pipeline_architecture.md
```

(If you downloaded them elsewhere, adjust the path above.)

### Step 6.2 — Update the script with your actual GitHub username

```bash
# Open in any editor — replace YOUR_USERNAME with your actual GitHub username
# The script doesn't need this explicitly but your README should link correctly

# Verify remote URLs are correct
cd ~/Documents/colonial-map-pipeline && git remote -v
cd ~/Documents/phd-vault && git remote -v
```

### Step 6.3 — Commit the new files

```bash
# Commit pipeline docs
cd ~/Documents/colonial-map-pipeline
git add docs/MASTER_PROCESS_FLOW.md
git commit -m "[E][20260507] Add master process flow doc

Obsidian-ref: VAULT_INIT
Notion-page: 35682b18
Next: Begin Script 01 — map ingestion
Blockers: none"
git push origin main

# Commit vault docs
cd ~/Documents/phd-vault
git add methodology/pipeline_architecture.md
git commit -m "Add pipeline architecture mirror from colonial-map-pipeline docs"
git push origin main
```

### Step 6.4 — Add a shell alias so you never forget the path

```bash
# Add to your shell config (~/.zshrc for modern Mac)
echo '' >> ~/.zshrc
echo '# PhD Research shortcuts' >> ~/.zshrc
echo 'alias phd-start="bash ~/Documents/phd_setup.sh start"' >> ~/.zshrc
echo 'alias phd-end="bash ~/Documents/phd_setup.sh end"' >> ~/.zshrc
echo 'alias phd-vault="cd ~/Documents/phd-vault"' >> ~/.zshrc
echo 'alias phd-repo="cd ~/Documents/colonial-map-pipeline"' >> ~/.zshrc

# Reload shell
source ~/.zshrc
```

Now instead of `bash ~/Documents/phd_setup.sh start` you just type:
```bash
phd-start
phd-end
```

---

## PHASE 7 — VERIFY EVERYTHING IS CONNECTED

Run this full verification sequence:

```bash
echo "=== GIT REPOS ==="
cd ~/Documents/colonial-map-pipeline && git log --oneline -3 && git remote -v
echo "---"
cd ~/Documents/phd-vault && git log --oneline -3 && git remote -v

echo ""
echo "=== VAULT STRUCTURE ==="
ls ~/Documents/phd-vault/

echo ""
echo "=== REPO STRUCTURE ==="
ls ~/Documents/colonial-map-pipeline/

echo ""
echo "=== ALIASES ==="
type phd-start
type phd-end
```

Expected output (roughly):
```
=== GIT REPOS ===
abc1234 [E][20260507] Add master process flow doc
def5678 [E][20260507] INIT: Scaffold colonial-map-pipeline repo
origin  https://github.com/YOUR_USERNAME/colonial-map-pipeline.git (fetch)
origin  https://github.com/YOUR_USERNAME/colonial-map-pipeline.git (push)
---
xyz9012 Add pipeline architecture mirror
uvw3456 [E][20260507] INIT: Scaffold phd-vault
origin  https://github.com/YOUR_USERNAME/phd-vault.git (fetch)
...

=== VAULT STRUCTURE ===
00_INDEX.md  admin  code  data  handoffs  literature  methodology  theory

=== REPO STRUCTURE ===
README.md  data  docs  notebooks  outputs  scripts  tests

=== ALIASES ===
phd-start: aliased to bash ~/Documents/phd_setup.sh start
phd-end: aliased to bash ~/Documents/phd_setup.sh end
```

If anything is missing, go back to the relevant phase and re-run that step.

---

## PHASE 8 — YOUR FIRST REAL SESSION

This is what you do right now, today, to start working.

```bash
# 1. Launch session
phd-start
# → Shows git log, shows LATEST_HANDOFF (will say "no handoff file" — that's OK for first session)
# → Asks session type: enter E (Planning)
# → Opens Notion in browser

# 2. In Obsidian: create your first handoff
# → ⌘P → "New note" → name it: 20260507_handoff_session-zero
# → ⌘P → "Insert template" → handoff_template
# → Fill in: session_type: E
# → Under "What Was Done": "Scaffolded full research environment"
# → Under "Next Session Must Start With": 
#     1. git checkout -b feat/map-ingestion
#     2. Begin writing scripts/01_map_ingestion.py
#     3. Test with one sample map from UCL OD

# 3. Create the LATEST_HANDOFF symlink (Terminal)
ln -sf ~/Documents/phd-vault/handoffs/20260507_handoff_session-zero.md \
        ~/Documents/phd-vault/handoffs/LATEST_HANDOFF.md

# 4. End session
phd-end
# → Prompts slug: type "session-zero"
# → Opens handoff in editor — fill in next_action field
# → Prompts commit message: "Initial environment setup complete"
# → Prompts next step: "Begin Script 01 map ingestion"
# → Prompts push: y
```

---

## QUICK REFERENCE — DAILY USE

```
EVERY SESSION START:
  phd-start

EVERY SESSION END:
  phd-end

CHECK GITHUB REPOS:
  github.com/YOUR_USERNAME/colonial-map-pipeline
  github.com/YOUR_USERNAME/phd-vault

CHECK OBSIDIAN SYNC STATUS:
  Bottom-right of Obsidian window → cloud icon

MANUAL VAULT GIT PUSH (if needed):
  cd ~/Documents/phd-vault && git add . && git commit -m "manual: vault update" && git push origin main

IF OBSIDIAN GIT PLUGIN FAILS:
  → Above manual command is your fallback
  → Obsidian Sync still has cloud copy regardless

IF OBSIDIAN SYNC LAPSES (subscription expired):
  → phd-vault GitHub repo still has everything
  → Git log is the complete history
```

---

## WHAT BACKS UP WHAT

| Layer | What | Where | Frequency | Recovery |
|-------|------|--------|-----------|----------|
| Git (vault) | All Obsidian notes + handoffs | GitHub: phd-vault | Every 30 min (auto) + every session end (manual) | `git clone` the repo |
| Obsidian Sync | All Obsidian notes + handoffs | Obsidian cloud servers | Real-time (seconds) | Log in to any device |
| Git (pipeline) | All scripts, docs, processed data | GitHub: colonial-map-pipeline | Every session end | `git clone` the repo |
| UCL OneDrive | Raw map TIFFs, raw scans | UCL OD at /Users/xon/UCL-OD/ | UCL IT managed | UCL IT recovery |

**Combined result:** You would need GitHub + Obsidian servers + UCL OneDrive to all fail simultaneously to lose work. That is not going to happen.

---

*Document saved at: `~/Documents/colonial-map-pipeline/docs/SETUP_GUIDE.md`*  
*Mirror at: `~/Documents/phd-vault/admin/SETUP_GUIDE.md`*

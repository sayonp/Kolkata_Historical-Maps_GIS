# Colonial Map Pipeline

**PI:** Spandita Pramanik | UCL Bartlett Space Syntax Lab  
**Research:** Spatio-Cultural Dynamics: Mapping Kolkata's Historical Processional Routes

## Quick Links
- **Notion Hub:** https://notion.so/35682b180df181c2b9cfdcba5718ab76
- **Obsidian Vault:** ~/Documents/phd-vault
- **Latest Handoff:** ~/Documents/phd-vault/handoffs/LATEST_HANDOFF.md
- **Master Process Flow:** _planning/PhD_MASTER_PROCESS_FLOW.md

## Session Commands
```bash
bash phd_setup.sh start   # cold-start every session
bash phd_setup.sh end     # handoff + commit at end
```

## Structure
```
cities/
  calcutta/          ← Kolkata-specific data, config, outputs
scripts/             ← 6 pipeline stages (city-aware)
_planning/           ← master plan docs (working reference)
tests/               ← pytest suite + tests.json tracker
docs/                ← pipeline spec, GCP guide
qgis/                ← QGIS project files
```

## Pipeline Stages
| Script | Stage | Status |
|--------|-------|--------|
| 01_map_ingestion.py | Standardise raw scans → GeoTIFF | trial — needs path review |
| 02_georeferencing.py | GDAL TPS georeferencing | no GCPs entered yet |
| 03_nlp_extraction.py | spaCy NLP: procession route extraction | stub |
| 04_route_mapping.py | Feature extraction → GeoJSON routes | trial — Windows paths |
| 05_syntax_analysis.py | Space syntax (DepthMapX + correlation) | stub |
| 06_abm_simulation.py | Mesa ABM swarm simulation | stub |

## Cities
- `calcutta` — active (Kolkata, colonial-era processional routes)
- *(additional cities added here as scope expands)*

## Data
Raw map TIFFs live on UCL OneDrive, not in this repo.  
Processed outputs (georeferenced GeoTIFFs, route GeoJSON) go in `cities/calcutta/data/processed/`.  
Git LFS decision pending before any processed TIFFs are committed.

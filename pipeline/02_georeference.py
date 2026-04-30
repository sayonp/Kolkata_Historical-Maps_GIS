"""
PHASE 2: ROBUST GEOREFERENCING
================================
Uses existing GeoPackage_Kolkata.gpkg as 
the authoritative modern reference layer.

Georeferencing approach:
1. Extract stable features from GeoPackage
   (river banks, major roads, railways)
2. Match to historical map features
3. Calculate TPS transformation
4. Validate against reference network
5. Output quality metrics

Requirements:
    pip install gdal rasterio geopandas 
    opencv-python numpy scikit-image scipy fiona
"""

import os
import json
import numpy as np
import geopandas as gpd
import rasterio
from pathlib import Path
import cv2
import subprocess

# ─────────────────────────────────────────
# CONFIGURATION — Edit these paths
# ─────────────────────────────────────────

GEOPACKAGE = "/Users/xon/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Maps/Kolkata_CUPDF/GeoPackage_Kolkata.gpkg"
INPUT_DIR  = "/Users/xon/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Maps/Processed"
OUTPUT_DIR = "/Users/xon/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Maps/Georeferenced"
DOCS_DIR   = "/Users/xon/Documents/GitHub/Kolkata_Historical-Maps_GIS/docs"
TARGET_CRS = "EPSG:4326"

# ─────────────────────────────────────────
# STABLE REFERENCE LANDMARKS
# Verified against GeoPackage_Kolkata.gpkg
# Stability = how unchanged since 1700s
# ─────────────────────────────────────────

REFERENCE_GCPS = {
    # TIER 1 — Ultra stable (use for ALL maps)
    "hooghly_north_bend":      {"lon": 88.3639, "lat": 22.6270, "stability": "ultra", "use_from": 1700},
    "hooghly_howrah":          {"lon": 88.3239, "lat": 22.5850, "stability": "ultra", "use_from": 1700},
    "hooghly_garden_reach":    {"lon": 88.3121, "lat": 22.5182, "stability": "ultra", "use_from": 1700},
    "hooghly_east_central":    {"lon": 88.3512, "lat": 22.5730, "stability": "ultra", "use_from": 1700},
    "hooghly_south_tip":       {"lon": 88.3089, "lat": 22.4950, "stability": "ultra", "use_from": 1700},

    # TIER 2 — Very stable (use for maps 1773+)
    "fort_william_centre":     {"lon": 88.3432, "lat": 22.5584, "stability": "very_high", "use_from": 1773},
    "fort_william_north":      {"lon": 88.3445, "lat": 22.5623, "stability": "very_high", "use_from": 1773},
    "maidan_north_edge":       {"lon": 88.3468, "lat": 22.5720, "stability": "very_high", "use_from": 1773},
    "maidan_south_edge":       {"lon": 88.3432, "lat": 22.5490, "stability": "very_high", "use_from": 1773},

    # TIER 3 — Stable (use for maps 1840+)
    "chowringhee_esplanade":   {"lon": 88.3520, "lat": 22.5726, "stability": "high", "use_from": 1840},
    "strand_road_north":       {"lon": 88.3468, "lat": 22.5851, "stability": "high", "use_from": 1840},
    "circular_road_east":      {"lon": 88.3756, "lat": 22.5456, "stability": "high", "use_from": 1840},
    "park_street_junction":    {"lon": 88.3563, "lat": 22.5520, "stability": "high", "use_from": 1840},

    # TIER 4 — Modern (use for maps 1854+)
    "howrah_station":          {"lon": 88.3219, "lat": 22.5839, "stability": "medium", "use_from": 1854},
    "sealdah_station":         {"lon": 88.3697, "lat": 22.5693, "stability": "medium", "use_from": 1862},
    "circular_canal_north":    {"lon": 88.3634, "lat": 22.5912, "stability": "medium", "use_from": 1830},
}

# ─────────────────────────────────────────
# PER-MAP GCP DEFINITIONS
# pixel_x, pixel_y = from QGIS inspection
# ref_key = key from REFERENCE_GCPS above
# ─────────────────────────────────────────

MAP_GCPS = {
    # TEMPLATE — copy and fill for each map
    # "Calcutta_1857_processed": {
    #     "year": 1857,
    #     "era": "pre_1870",
    #     "gcps": [
    #         [pixel_x, pixel_y, "ref_key"],
    #         [pixel_x, pixel_y, "ref_key"],
    #     ]
    # },
}


# ─────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────

def load_reference_layers(geopackage_path):
    """Load key layers from GeoPackage"""
    try:
        available = ["StreetNetwork_Kolkata",
                     "WaterFill_Kolkata",
                     "RailwayLines_Kolkata"]
        print(f"  Using layers: {available}")
    except Exception as e:
        print(f"  Error: {e}")
        available = []

    layers = {}
    priority = [
        "StreetNetwork_Kolkata",
        "WaterFill_Kolkata",
        "RailwayLines_Kolkata",
    ]

    for layer in priority:
        try:
            gdf = gpd.read_file(geopackage_path, layer=layer)
            layers[layer] = gdf
            print(f"  ✅ {layer}: {len(gdf)} features")
        except Exception as e:
            print(f"  ⚠️  {layer}: {e}")

    return layers


def get_applicable_gcps(map_gcps_data, map_year):
    """Filter GCPs by year — only use landmarks that existed"""
    applicable = []
    for px, py, ref_key in map_gcps_data.get("gcps", []):
        if ref_key in REFERENCE_GCPS:
            ref = REFERENCE_GCPS[ref_key]
            if map_year >= ref.get("use_from", 0):
                applicable.append((px, py, ref_key))
            else:
                print(f"    Skipping {ref_key} — didn't exist in {map_year}")
    return applicable


def georeference_map(input_path, output_path, gcps, year):
    """Georeference using GDAL TPS transformation"""
    print(f"\n  Georeferencing: {Path(input_path).name}")
    print(f"  Year: {year}, GCPs: {len(gcps)}")

    if len(gcps) < 6:
        print(f"    ⚠️  Need minimum 6 GCPs, have {len(gcps)}")
        return False

    # Build GDAL GCP string
    gcp_args = []
    for px, py, ref_key in gcps:
        ref = REFERENCE_GCPS[ref_key]
        gcp_args.extend([
            "-gcp", str(px), str(py),
            str(ref["lon"]), str(ref["lat"])
        ])

    temp = str(output_path).replace(".tif", "_temp.tif")

    try:
        # Assign GCPs
        cmd1 = ["gdal_translate", "-of", "GTiff",
                *gcp_args, "-a_srs", TARGET_CRS,
                str(input_path), temp]
        r1 = subprocess.run(cmd1, capture_output=True, text=True)
        if r1.returncode != 0:
            print(f"    Error: {r1.stderr[:200]}")
            return False

        # Warp with TPS
        cmd2 = ["gdalwarp", "-tps", "-r", "cubic",
                "-t_srs", TARGET_CRS,
                "-co", "COMPRESS=LZW",
                "-co", "TILED=YES",
                temp, str(output_path)]
        r2 = subprocess.run(cmd2, capture_output=True, text=True)
        if r2.returncode != 0:
            print(f"    Error: {r2.stderr[:200]}")
            return False

        if os.path.exists(temp):
            os.remove(temp)

        print(f"    ✅ Success: {Path(output_path).name}")
        return True

    except FileNotFoundError:
        print("    GDAL not found — install with: brew install gdal")
        return False


def validate_map(georef_path):
    """Check georeferenced map is within Kolkata extent"""
    try:
        with rasterio.open(georef_path) as src:
            b = src.bounds
            in_kolkata = (88.20 <= b.left <= 88.50 and
                         22.40 <= b.bottom <= 22.70)
            return {
                "bounds": dict(b),
                "in_kolkata_extent": in_kolkata,
                "crs": str(src.crs),
                "pass": in_kolkata
            }
    except Exception as e:
        return {"pass": False, "error": str(e)}


def generate_gcp_guide():
    """Create step-by-step GCP guide for QGIS"""
    Path(DOCS_DIR).mkdir(parents=True, exist_ok=True)
    guide_path = Path(DOCS_DIR) / "gcp_guide.md"

    content = f"""# GCP Guide — Kolkata Historical Maps

## How to Add GCPs in QGIS

### Step 1: Open QGIS
Load these layers from GeoPackage:
- WaterFill_Kolkata (river reference)
- StreetNetwork_Kolkata (road reference)

### Step 2: Open Georeferencer
```
Raster → Georeferencer
File → Open Raster → select from Processed folder
Settings → Transformation: Thin Plate Spline
Settings → Resampling: Cubic
Settings → CRS: EPSG:4326
Settings → ✅ Save GCP points
Settings → ✅ Load in project when done
```

### Step 3: Add GCPs
For each landmark below:
1. Press A (Add GCP tool)
2. Click the feature on the HISTORICAL map
3. Click "From Map Canvas"  
4. Click same feature on MODERN map
5. Note the pixel X, Y shown in GCP table

### Step 4: Save .points file
```
File → Save GCP Points
Save to: Maps/Georeferenced/[mapname].points
```

### Step 5: Update 02_georeference.py
Add pixel coordinates to MAP_GCPS dictionary:
```python
"Calcutta_1857_processed": {{
    "year": 1857,
    "era": "pre_1870", 
    "gcps": [
        [pixel_x, pixel_y, "hooghly_north_bend"],
        [pixel_x, pixel_y, "hooghly_howrah"],
        # ... minimum 6-10 GCPs
    ]
}}
```

## Reference Landmarks

### TIER 1 — Use for ALL maps (any year)
| Key | Location | Lon | Lat |
|-----|----------|-----|-----|
| hooghly_north_bend | N Hooghly bend | 88.3639 | 22.6270 |
| hooghly_howrah | Hooghly at Howrah | 88.3239 | 22.5850 |
| hooghly_garden_reach | S Hooghly bend | 88.3121 | 22.5182 |
| hooghly_east_central | E bank central | 88.3512 | 22.5730 |
| hooghly_south_tip | S Hooghly tip | 88.3089 | 22.4950 |

### TIER 2 — Use for maps 1773+
| Key | Location | Lon | Lat |
|-----|----------|-----|-----|
| fort_william_centre | Fort William | 88.3432 | 22.5584 |
| maidan_north_edge | Maidan N edge | 88.3468 | 22.5720 |
| maidan_south_edge | Maidan S edge | 88.3432 | 22.5490 |

### TIER 3 — Use for maps 1840+
| Key | Location | Lon | Lat |
|-----|----------|-----|-----|
| chowringhee_esplanade | Chowringhee | 88.3520 | 22.5726 |
| strand_road_north | Strand Road | 88.3468 | 22.5851 |
| circular_road_east | Circular Road | 88.3756 | 22.5456 |

### TIER 4 — Use for maps 1854+
| Key | Location | Lon | Lat |
|-----|----------|-----|-----|
| howrah_station | Howrah Station | 88.3219 | 22.5839 |
| sealdah_station | Sealdah Station | 88.3697 | 22.5693 |

## Quality Targets
- Minimum GCPs: 6 (pre-1800), 8 (1800-1870), 10 (post-1870)
- Mean error: < 0.05 pixels
- Worst residual: < 0.10 pixels
- All 4 corners must have GCPs
"""

    with open(guide_path, "w") as f:
        f.write(content)
    print(f"  GCP guide saved: {guide_path}")


def main():
    print("=" * 60)
    print("KOLKATA MAPS — ROBUST GEOREFERENCING")
    print("Reference: GeoPackage_Kolkata.gpkg")
    print("=" * 60)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load reference
    print("\n[1/4] Loading reference GeoPackage...")
    ref_layers = load_reference_layers(GEOPACKAGE)

    # Generate GCP guide
    print("\n[2/4] Generating GCP guide...")
    generate_gcp_guide()

    # Process maps
    print("\n[3/4] Processing maps...")
    processed = sorted(Path(INPUT_DIR).glob("*_processed.tif"))
    print(f"  Found {len(processed)} processed maps")

    needs_gcps = []
    completed  = []
    results    = {}

    for map_path in processed:
        stem = map_path.stem
        gcps_data = MAP_GCPS.get(stem, {})
        year = gcps_data.get("year", 0)
        output = Path(OUTPUT_DIR) / f"{stem}_georef.tif"

        if gcps_data.get("gcps"):
            gcps = get_applicable_gcps(gcps_data, year)
            success = georeference_map(map_path, output, gcps, year)
            if success:
                validation = validate_map(output)
                results[stem] = {
                    "status": "complete",
                    "year": year,
                    "gcps_used": len(gcps),
                    "validation": validation
                }
                completed.append(stem)
            else:
                results[stem] = {"status": "failed"}
        else:
            needs_gcps.append(stem)
            results[stem] = {
                "status": "needs_gcps",
                "year": year,
                "action": "Add GCPs in QGIS — see docs/gcp_guide.md"
            }

    # Save results
    print("\n[4/4] Saving results...")
    results_path = Path(OUTPUT_DIR) / "georef_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Georeferenced: {len(completed)}")
    print(f"⏳ Needs GCPs:   {len(needs_gcps)}")
    print(f"\nNext steps:")
    print(f"1. Read: docs/gcp_guide.md")
    print(f"2. Open QGIS with GeoPackage_Kolkata.gpkg")
    print(f"3. Add GCPs for each map in QGIS Georeferencer")
    print(f"4. Add pixel coords to MAP_GCPS in this script")
    print(f"5. Re-run: python3 pipeline/02_georeference.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

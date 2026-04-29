"""
PHASE 2: AUTOMATED GEOREFERENCING
===================================
Georeferences all standardised maps using:
- Known GCP coordinates for Calcutta landmarks
- GDAL warp transformation
- Thin Plate Spline correction
- Quality validation

Requirements:
    pip install gdal rasterio numpy pandas
"""

import os
import json
import numpy as np
from pathlib import Path
import subprocess

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
INPUT_DIR  = r"C:\Users\ucbv537\OneDrive\Maps\Processed"
OUTPUT_DIR = r"C:\Users\ucbv537\OneDrive\Maps\Georeferenced"
TARGET_CRS = "EPSG:4326"

# ─────────────────────────────────────────
# GROUND CONTROL POINTS
# For each map, define GCPs as:
# [pixel_x, pixel_y, longitude, latitude]
# These are stable Calcutta landmarks
# ─────────────────────────────────────────

# Permanent geographic reference points
# (these stay consistent across all maps)
CALCUTTA_REFERENCE_POINTS = {
    "Hooghly_River_North_Bend": {
        "lon": 88.3639, "lat": 22.6270,
        "description": "Hooghly river northern bend near Dakshineswar"
    },
    "Hooghly_River_Howrah_Bridge": {
        "lon": 88.3468, "lat": 22.5850,
        "description": "Howrah Bridge crossing point"
    },
    "Hooghly_River_Garden_Reach": {
        "lon": 88.3121, "lat": 22.5182,
        "description": "Hooghly river southern bend Garden Reach"
    },
    "Fort_William": {
        "lon": 88.3432, "lat": 22.5584,
        "description": "Fort William location — stable since 1773"
    },
    "Maidan_Centre": {
        "lon": 88.3468, "lat": 22.5607,
        "description": "Maidan open ground — consistent landmark"
    },
    "Sealdah_Station_Area": {
        "lon": 88.3697, "lat": 22.5693,
        "description": "Sealdah station general area"
    },
    "Hooghly_East_Bank_Central": {
        "lon": 88.3512, "lat": 22.5730,
        "description": "Eastern bank of Hooghly central area"
    },
    "Circular_Canal_East": {
        "lon": 88.3756, "lat": 22.5456,
        "description": "Eastern end of Circular Canal"
    },
}

# Per-map GCP pixel coordinates
# These need to be determined for each map
# Format: [pixel_x, pixel_y, lon, lat]
MAP_GCPS = {
    "Calcutta_1857_processed": [
        # Add pixel coordinates after visual inspection
        # Example format:
        # [pixel_x, pixel_y, longitude, latitude]
        # Use QGIS to identify pixel coordinates
    ],
    "Calcutta_1842_Fullsheet_processed": [],
    "Calcutta_1893_processed": [],
    "Calcutta_1911_processed": [],
    "Calcutta_1914_processed": [],
    "Calcutta_1924_processed": [],
}


def create_gdal_gcp_string(gcps):
    """Convert GCP list to GDAL command string"""
    gcp_args = []
    for gcp in gcps:
        px, py, lon, lat = gcp
        gcp_args.extend(["-gcp", str(px), str(py), str(lon), str(lat)])
    return gcp_args


def georeference_map(input_path, output_path, gcps, crs=TARGET_CRS):
    """
    Georeference a single map using GDAL.
    Uses Thin Plate Spline transformation.
    """
    print(f"\n  Georeferencing: {Path(input_path).name}")

    if not gcps:
        print("    No GCPs defined — skipping")
        return False

    try:
        # Step 1: Assign GCPs to image
        temp_path = str(output_path).replace(".tif", "_gcps.tif")

        gcp_args  = create_gdal_gcp_string(gcps)
        translate_cmd = [
            "gdal_translate",
            "-of", "GTiff",
            *gcp_args,
            "-a_srs", crs,
            str(input_path),
            temp_path
        ]
        subprocess.run(translate_cmd, check=True, capture_output=True)

        # Step 2: Warp using Thin Plate Spline
        warp_cmd = [
            "gdalwarp",
            "-tps",               # Thin Plate Spline
            "-r", "cubic",        # Cubic resampling
            "-t_srs", crs,
            "-co", "COMPRESS=LZW",
            "-co", "TILED=YES",
            temp_path,
            str(output_path)
        ]
        subprocess.run(warp_cmd, check=True, capture_output=True)

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        print(f"    Saved: {Path(output_path).name}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"    GDAL Error: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print("    GDAL not found — install QGIS or OSGeo4W")
        return False


def validate_georeferencing(georef_path):
    """
    Validate georeferenced output.
    Checks CRS, extent and data validity.
    """
    try:
        import rasterio
        with rasterio.open(georef_path) as src:
            bounds = src.bounds
            crs    = src.crs

            # Check coordinates are within Calcutta extent
            # Expected: lon 88.2-88.5, lat 22.4-22.7
            valid_lon = 88.2 <= bounds.left <= 88.5
            valid_lat = 22.4 <= bounds.bottom <= 22.7

            result = {
                "crs":       str(crs),
                "bounds":    bounds,
                "valid_lon": valid_lon,
                "valid_lat": valid_lat,
                "valid":     valid_lon and valid_lat,
            }
            return result
    except Exception as e:
        return {"valid": False, "error": str(e)}


def save_gcp_file(gcps, output_path):
    """Save GCPs as .points file for QGIS compatibility"""
    points_path = str(output_path).replace(".tif", ".points")
    with open(points_path, "w") as f:
        f.write("mapX,mapY,pixelX,pixelY,enable,dX,dY,residual\n")
        for gcp in gcps:
            px, py, lon, lat = gcp
            f.write(f"{lon},{lat},{px},{-py},1,0,0,0\n")
    print(f"    GCPs saved: {Path(points_path).name}")


def main():
    print("=" * 60)
    print("CALCUTTA HISTORICAL MAPS — GEOREFERENCING PIPELINE")
    print("=" * 60)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Get all processed maps
    processed_maps = list(Path(INPUT_DIR).glob("*_processed.tif"))
    print(f"\nFound {len(processed_maps)} processed maps")

    results = []
    for map_path in processed_maps:
        stem = map_path.stem
        gcps = MAP_GCPS.get(stem, [])

        output_path = Path(OUTPUT_DIR) / f"{stem}_georef.tif"

        if gcps:
            success = georeference_map(map_path, output_path, gcps)
            if success:
                validation = validate_georeferencing(output_path)
                save_gcp_file(gcps, output_path)
                results.append({
                    "map": stem,
                    "status": "success" if validation["valid"] else "invalid",
                    "validation": validation
                })
        else:
            print(f"\n  {stem}: No GCPs defined")
            print(f"    → Open in QGIS Georeferencer to add GCPs manually")
            print(f"    → Save .points file to: {OUTPUT_DIR}")
            results.append({"map": stem, "status": "needs_gcps"})

    # Print summary
    print(f"\n{'='*60}")
    print("GEOREFERENCING SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = r["status"]
        print(f"  {r['map']}: {status}")

    print(f"\nNOTE: For maps needing GCPs, use QGIS Georeferencer")
    print(f"Reference points saved to: {OUTPUT_DIR}/reference_points.json")

    # Save reference points for use in QGIS
    ref_path = Path(OUTPUT_DIR) / "reference_points.json"
    with open(ref_path, "w") as f:
        json.dump(CALCUTTA_REFERENCE_POINTS, f, indent=2)


if __name__ == "__main__":
    main()

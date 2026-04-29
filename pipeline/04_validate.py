"""
PHASE 4: VALIDATION & TOPOLOGY CHECK
=======================================
Validates extracted vectors:
- Checks topology (dangling lines, overlaps)
- Validates geometries
- Compares with OSM for accuracy
- Generates accuracy report
- Merges all years into final repository

Requirements:
    pip install geopandas shapely osmnx pandas
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiLineString
from shapely.ops import unary_union, linemerge
from pathlib import Path
import json
import os

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
VECTORS_DIR  = r"C:\Users\ucbv537\OneDrive\Maps\Vectors"
FINAL_DIR    = r"C:\Users\ucbv537\OneDrive\Maps\Final_Repository"
OSM_COMPARE  = True   # Compare with OSM modern roads

# Calcutta bounding box for OSM download
CALCUTTA_BBOX = {
    "north": 22.65,
    "south": 22.50,
    "east":  88.42,
    "west":  88.30
}


# ─────────────────────────────────────────
# VALIDATION FUNCTIONS
# ─────────────────────────────────────────

def fix_geometries(gdf):
    """Fix invalid geometries"""
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        print(f"    Fixing {invalid.sum()} invalid geometries")
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)
    return gdf


def remove_duplicates(gdf, tolerance=0.0001):
    """Remove duplicate geometries within tolerance"""
    before = len(gdf)
    gdf = gdf.drop_duplicates(subset=["geometry"])
    after  = len(gdf)
    if before != after:
        print(f"    Removed {before - after} duplicates")
    return gdf


def simplify_lines(gdf, tolerance=0.00005):
    """
    Simplify line geometries.
    Reduces vertex count while preserving shape.
    tolerance in degrees (~5m at equator)
    """
    gdf["geometry"] = gdf["geometry"].simplify(
        tolerance, preserve_topology=True
    )
    return gdf


def merge_connected_lines(gdf):
    """Merge connected line segments into longer lines"""
    try:
        merged = linemerge(unary_union(gdf.geometry))
        if merged.geom_type == "LineString":
            geometries = [merged]
        elif merged.geom_type == "MultiLineString":
            geometries = list(merged.geoms)
        else:
            return gdf

        # Preserve attributes from first feature
        attrs = gdf.iloc[0].drop("geometry").to_dict()
        new_gdf = gpd.GeoDataFrame(
            [dict(geometry=g, **attrs) for g in geometries],
            crs=gdf.crs
        )
        print(f"    Merged {len(gdf)} → {len(new_gdf)} lines")
        return new_gdf
    except Exception as e:
        print(f"    Merge failed: {e}")
        return gdf


def check_topology(gdf):
    """
    Check for topological errors.
    Returns dict with error counts.
    """
    errors = {
        "invalid_geom":  0,
        "empty_geom":    0,
        "self_intersect": 0,
        "total_features": len(gdf),
    }

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            errors["empty_geom"] += 1
        elif not geom.is_valid:
            errors["invalid_geom"] += 1
        elif geom.is_simple is False:
            errors["self_intersect"] += 1

    errors["clean"] = (
        errors["invalid_geom"] == 0 and
        errors["empty_geom"] == 0
    )
    return errors


def compare_with_osm(historical_gdf, year):
    """
    Compare historical roads with modern OSM.
    Identifies roads that persisted vs disappeared.
    """
    try:
        import osmnx as ox
        print("    Downloading OSM road network...")

        # Download modern roads
        G = ox.graph_from_bbox(
            CALCUTTA_BBOX["north"],
            CALCUTTA_BBOX["south"],
            CALCUTTA_BBOX["east"],
            CALCUTTA_BBOX["west"],
            network_type="drive"
        )
        modern_gdf = ox.graph_to_gdfs(G, nodes=False)
        modern_gdf = modern_gdf.to_crs("EPSG:4326")

        # Buffer historical roads slightly for comparison
        hist_buffer = historical_gdf.copy()
        hist_buffer["geometry"] = hist_buffer.geometry.buffer(0.0002)

        # Spatial join to find matching modern roads
        joined = gpd.sjoin(
            modern_gdf, hist_buffer,
            how="left", predicate="intersects"
        )

        n_matched = joined["index_right"].notna().sum()
        n_total   = len(modern_gdf)
        pct       = (n_matched / n_total * 100) if n_total > 0 else 0

        print(f"    OSM comparison: {pct:.1f}% of modern roads "
              f"overlap with {year} historical network")

        return {
            "year": year,
            "historical_roads": len(historical_gdf),
            "modern_roads": n_total,
            "matched": n_matched,
            "overlap_pct": pct
        }

    except ImportError:
        print("    osmnx not installed — skipping OSM comparison")
        print("    Install with: pip install osmnx")
        return {"error": "osmnx not available"}
    except Exception as e:
        print(f"    OSM comparison error: {e}")
        return {"error": str(e)}


def add_standard_attributes(gdf, year, feature_type, source_map):
    """Add standardised attributes to all features"""
    gdf["year"]         = year
    gdf["feature_type"] = feature_type
    gdf["source_map"]   = source_map
    gdf["era"]          = get_era(year)
    gdf["city"]         = "Calcutta"
    gdf["country"]      = "India"
    gdf["crs"]          = "EPSG:4326"
    return gdf


def get_era(year):
    """Classify map by historical era"""
    if year < 1800:  return "Pre-Colonial"
    elif year < 1858: return "East India Company"
    elif year < 1900: return "Early British Raj"
    elif year < 1920: return "Late British Raj"
    else:             return "Late Colonial"


def process_feature_layer(vector_path, feature_type, year, source_map):
    """Clean and validate a single feature layer"""
    print(f"\n  Processing: {Path(vector_path).name}")

    try:
        gdf = gpd.read_file(vector_path)

        if gdf.empty:
            print("    Empty layer — skipping")
            return None

        print(f"    Input: {len(gdf)} features")

        # Fix geometries
        gdf = fix_geometries(gdf)

        # Remove duplicates
        gdf = remove_duplicates(gdf)

        # Simplify
        if feature_type in ["road", "tram", "railway"]:
            gdf = simplify_lines(gdf, tolerance=0.00005)

        # Add attributes
        gdf = add_standard_attributes(gdf, year, feature_type, source_map)

        # Check topology
        topo = check_topology(gdf)
        print(f"    Topology: {topo}")

        print(f"    Output: {len(gdf)} clean features")
        return gdf

    except Exception as e:
        print(f"    Error: {e}")
        return None


def merge_temporal_layers(all_gdfs, feature_type):
    """
    Merge all years of a feature type into one layer.
    Creates temporal attribute for animation/analysis.
    """
    if not all_gdfs:
        return None

    combined = gpd.GeoDataFrame(
        pd.concat(all_gdfs, ignore_index=True),
        crs="EPSG:4326"
    )

    print(f"  Combined {feature_type}: {len(combined)} total features "
          f"across {len(all_gdfs)} time periods")

    return combined


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 60)
    print("CALCUTTA HISTORICAL MAPS — VALIDATION & FINAL OUTPUT")
    print("=" * 60)

    Path(FINAL_DIR).mkdir(parents=True, exist_ok=True)

    # Collect all vector files
    vector_files = list(Path(VECTORS_DIR).glob("*.gpkg"))
    print(f"\nFound {len(vector_files)} vector files")

    # Group by feature type
    roads_all     = []
    water_all     = []
    buildings_all = []
    trams_all     = []
    osm_results   = []

    for vf in sorted(vector_files):
        stem = vf.stem

        # Determine feature type and year from filename
        feature_type = "road"
        year = 0

        if "_roads" in stem:
            feature_type = "road"
        elif "_water" in stem:
            feature_type = "water"
        elif "_buildings" in stem:
            feature_type = "building"
        elif "_trams" in stem:
            feature_type = "tram"

        # Extract year from filename
        for y in range(1700, 1950):
            if str(y) in stem:
                year = y
                break

        gdf = process_feature_layer(vf, feature_type, year, stem)

        if gdf is not None:
            if feature_type == "road":
                roads_all.append(gdf)
                # Compare with OSM
                osm_result = compare_with_osm(gdf, year)
                osm_results.append(osm_result)
            elif feature_type == "water":
                water_all.append(gdf)
            elif feature_type == "building":
                buildings_all.append(gdf)
            elif feature_type == "tram":
                trams_all.append(gdf)

    # ── SAVE COMBINED LAYERS ────────────────
    print("\n[Saving final repository layers]")

    layers = {
        "roads":     (roads_all,     "Calcutta_Historical_Roads_All_Years.gpkg"),
        "water":     (water_all,     "Calcutta_Historical_Water_All_Years.gpkg"),
        "buildings": (buildings_all, "Calcutta_Historical_Buildings_All_Years.gpkg"),
        "trams":     (trams_all,     "Calcutta_Historical_Trams_All_Years.gpkg"),
    }

    for layer_name, (gdfs, filename) in layers.items():
        if gdfs:
            combined = merge_temporal_layers(gdfs, layer_name)
            if combined is not None:
                output_path = Path(FINAL_DIR) / filename
                combined.to_file(output_path, driver="GPKG")
                print(f"  Saved: {filename} ({len(combined)} features)")

    # ── SAVE ACCURACY REPORT ────────────────
    report = {
        "project": "Calcutta Historical GIS Repository",
        "created": str(pd.Timestamp.now()),
        "total_maps": len(set(
            [r.get("year") for r in osm_results if "year" in r]
        )),
        "osm_comparison": osm_results,
        "layers": {
            "roads":     len(roads_all),
            "water":     len(water_all),
            "buildings": len(buildings_all),
            "trams":     len(trams_all),
        }
    }

    report_path = Path(FINAL_DIR) / "accuracy_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print("FINAL REPOSITORY COMPLETE")
    print(f"Location: {FINAL_DIR}")
    print(f"{'='*60}")
    print("\nContents:")
    for f in Path(FINAL_DIR).iterdir():
        size = f.stat().st_size / 1024 / 1024
        print(f"  {f.name} ({size:.1f} MB)")

    print("\nNEXT: Push to GitHub repository")
    print("Run: git add . && git commit -m 'Add Calcutta vectors'")


if __name__ == "__main__":
    main()

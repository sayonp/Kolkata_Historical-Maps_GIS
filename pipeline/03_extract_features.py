"""
PHASE 3: FEATURE EXTRACTION
==============================
Extracts GIS features from georeferenced historical maps:
- Road networks
- Water bodies (rivers, canals, tanks)
- Building footprints
- Railways

Uses colour segmentation + morphological operations.
Each map era requires colour profile tuning.

Requirements:
    pip install opencv-python numpy rasterio geopandas shapely scikit-image tqdm
"""

import cv2
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import unary_union
from rasterio.features import shapes
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from pathlib import Path
from tqdm import tqdm
import json

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
INPUT_DIR  = r"C:\Users\ucbv537\OneDrive\Maps\Georeferenced"
OUTPUT_DIR = r"C:\Users\ucbv537\OneDrive\Maps\Vectors"

# ─────────────────────────────────────────
# COLOUR PROFILES PER MAP ERA
# HSV colour ranges for feature detection
# Tune these per map based on visual inspection
# HSV: Hue(0-180), Saturation(0-255), Value(0-255)
# ─────────────────────────────────────────
COLOUR_PROFILES = {
    # Pre-1870 maps (typically black & white engraving)
    "pre_1870": {
        "roads": {
            "description": "Dark lines on light background",
            "method": "edge_detection",
            "canny_low": 50,
            "canny_high": 150,
            "min_line_length": 30,
            "max_line_gap": 10,
        },
        "water": {
            "description": "Hatched/stippled areas near river",
            "method": "texture",
            "threshold": 180,
        },
        "buildings": {
            "description": "Dense black fill areas",
            "method": "density",
            "threshold": 100,
        },
    },

    # 1870-1900 maps (often hand-coloured)
    "1870_1900": {
        "roads": {
            "description": "White/light lines or dark outlines",
            "method": "edge_detection",
            "canny_low": 40,
            "canny_high": 120,
            "min_line_length": 25,
            "max_line_gap": 15,
        },
        "water": {
            "description": "Blue colour",
            "method": "colour",
            "hsv_lower": np.array([90, 50, 50]),
            "hsv_upper": np.array([130, 255, 255]),
        },
        "buildings": {
            "description": "Pink/red colour (built-up areas)",
            "method": "colour",
            "hsv_lower": np.array([0, 50, 150]),
            "hsv_upper": np.array([20, 255, 255]),
        },
    },

    # 1900-1930 IGI style maps (your 1907/1911 maps)
    "1900_1930": {
        "roads": {
            "description": "White roads with dark outlines",
            "method": "combined",
            "canny_low": 30,
            "canny_high": 100,
            "min_line_length": 20,
            "max_line_gap": 20,
        },
        "tram_routes": {
            "description": "Bold red/orange lines",
            "method": "colour",
            "hsv_lower": np.array([0, 100, 100]),
            "hsv_upper": np.array([15, 255, 255]),
        },
        "water": {
            "description": "Blue/teal filled areas",
            "method": "colour",
            "hsv_lower": np.array([85, 40, 100]),
            "hsv_upper": np.array([135, 255, 255]),
        },
        "buildings": {
            "description": "Red/pink dense areas",
            "method": "colour",
            "hsv_lower": np.array([0, 60, 150]),
            "hsv_upper": np.array([20, 255, 255]),
        },
        "railways": {
            "description": "Black lines with tick marks",
            "method": "edge_detection",
            "canny_low": 80,
            "canny_high": 200,
            "min_line_length": 40,
            "max_line_gap": 5,
        },
    },
}

# Map each processed file to its colour profile
MAP_PROFILES = {
    "Calcutta_1735": "pre_1870",
    "Calcutta_1756": "pre_1870",
    "Calcutta_1832": "pre_1870",
    "Calcutta_1842": "pre_1870",
    "Calcutta_1847": "pre_1870",
    "Calcutta_1857": "pre_1870",
    "Calcutta_1865": "pre_1870",
    "Calcutta_1893": "1870_1900",
    "Calcutta_1894": "1870_1900",
    "Calcutta_1911": "1900_1930",
    "Calcutta_1912": "1900_1930",
    "Calcutta_1914": "1900_1930",
    "Calcutta_1924": "1900_1930",
}


# ─────────────────────────────────────────
# FEATURE EXTRACTION FUNCTIONS
# ─────────────────────────────────────────

def load_georeferenced_map(map_path):
    """Load georeferenced raster and return image + transform"""
    with rasterio.open(map_path) as src:
        img       = src.read([1, 2, 3])  # RGB bands
        transform = src.transform
        crs       = src.crs
        # Rearrange to HxWxC for OpenCV
        img = np.moveaxis(img, 0, -1)
    return img, transform, crs


def extract_roads_edge_detection(img, profile):
    """
    Extract roads using edge detection.
    Works well for black & white engraved maps.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Canny edge detection
    edges = cv2.Canny(
        blurred,
        profile["canny_low"],
        profile["canny_high"]
    )

    # Dilate edges slightly to connect broken lines
    kernel = np.ones((2, 2), np.uint8)
    edges  = cv2.dilate(edges, kernel, iterations=1)

    # Remove small noise
    binary = edges > 0
    binary = remove_small_objects(binary, min_size=50)

    return binary.astype(np.uint8) * 255


def extract_features_by_colour(img, hsv_lower, hsv_upper):
    """
    Extract features by colour range in HSV space.
    Works well for coloured maps (post-1870).
    """
    # Convert RGB to BGR for OpenCV
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Create colour mask
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    # Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def skeletonize_roads(binary_mask):
    """
    Reduce road areas to single-pixel centrelines.
    Essential for creating clean vector lines.
    """
    # Ensure binary
    binary = binary_mask > 0

    # Skeletonize
    skeleton = skeletonize(binary)

    return skeleton.astype(np.uint8) * 255


def raster_to_lines(skeleton, transform, min_length=10):
    """
    Convert skeleton raster to vector LineStrings.
    Returns GeoDataFrame with line geometries.
    """
    lines = []

    # Find connected components
    labeled = label(skeleton > 0)

    for region in regionprops(labeled):
        coords = region.coords  # pixel coordinates

        if len(coords) < min_length:
            continue

        # Convert pixel coords to geographic coords
        geo_coords = []
        for row, col in coords:
            x, y = rasterio.transform.xy(transform, row, col)
            geo_coords.append((x, y))

        if len(geo_coords) >= 2:
            try:
                line = LineString(geo_coords)
                lines.append(line)
            except Exception:
                pass

    if not lines:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
    return gdf


def raster_to_polygons(mask, transform, min_area=100):
    """
    Convert binary mask to vector polygons.
    Used for water bodies and buildings.
    """
    polygons = []

    mask_uint8 = mask.astype(np.uint8)

    for geom, val in shapes(mask_uint8, transform=transform):
        if val == 1:
            from shapely.geometry import shape
            poly = shape(geom)
            if poly.area > min_area * (transform.a ** 2):
                polygons.append(poly)

    if not polygons:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    merged = unary_union(polygons)
    if merged.geom_type == "Polygon":
        polygons = [merged]
    elif merged.geom_type == "MultiPolygon":
        polygons = list(merged.geoms)

    gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
    return gdf


def extract_all_features(map_path, profile_name, output_dir, map_year):
    """
    Main extraction function — extracts all features from one map.
    """
    print(f"\n  Extracting features: {Path(map_path).name}")

    profile = COLOUR_PROFILES.get(profile_name, COLOUR_PROFILES["pre_1870"])

    # Load map
    img, transform, crs = load_georeferenced_map(map_path)
    stem = Path(map_path).stem

    results = {}

    # ── ROADS ────────────────────────────
    print("    Extracting roads...")
    road_profile = profile.get("roads", {})
    method = road_profile.get("method", "edge_detection")

    if method in ["edge_detection", "combined"]:
        road_mask = extract_roads_edge_detection(img, road_profile)
    elif method == "colour":
        road_mask = extract_features_by_colour(
            img,
            road_profile["hsv_lower"],
            road_profile["hsv_upper"]
        )
    else:
        road_mask = extract_roads_edge_detection(img, road_profile)

    road_skeleton = skeletonize_roads(road_mask)
    roads_gdf = raster_to_lines(road_skeleton, transform)
    roads_gdf["year"]     = map_year
    roads_gdf["type"]     = "road"
    roads_gdf["source"]   = Path(map_path).stem
    roads_gdf["era"]      = profile_name

    road_output = Path(output_dir) / f"{stem}_roads.gpkg"
    if not roads_gdf.empty:
        roads_gdf.to_file(road_output, driver="GPKG")
        print(f"    Roads: {len(roads_gdf)} features → {road_output.name}")
    results["roads"] = str(road_output)

    # ── WATER BODIES ────────────────────────────
    water_profile = profile.get("water", {})
    if water_profile.get("method") == "colour":
        print("    Extracting water bodies...")
        water_mask = extract_features_by_colour(
            img,
            water_profile["hsv_lower"],
            water_profile["hsv_upper"]
        )
        water_gdf = raster_to_polygons(water_mask, transform)
        water_gdf["year"]   = map_year
        water_gdf["type"]   = "water"
        water_gdf["source"] = Path(map_path).stem

        water_output = Path(output_dir) / f"{stem}_water.gpkg"
        if not water_gdf.empty:
            water_gdf.to_file(water_output, driver="GPKG")
            print(f"    Water: {len(water_gdf)} features → {water_output.name}")
        results["water"] = str(water_output)

    # ── BUILDINGS ────────────────────────────
    building_profile = profile.get("buildings", {})
    if building_profile.get("method") == "colour":
        print("    Extracting buildings/built-up areas...")
        bld_mask = extract_features_by_colour(
            img,
            building_profile["hsv_lower"],
            building_profile["hsv_upper"]
        )
        bld_gdf = raster_to_polygons(bld_mask, transform)
        bld_gdf["year"]   = map_year
        bld_gdf["type"]   = "building"
        bld_gdf["source"] = Path(map_path).stem

        bld_output = Path(output_dir) / f"{stem}_buildings.gpkg"
        if not bld_gdf.empty:
            bld_gdf.to_file(bld_output, driver="GPKG")
            print(f"    Buildings: {len(bld_gdf)} features → {bld_output.name}")
        results["buildings"] = str(bld_output)

    # ── TRAM ROUTES (1900+ maps only) ────────────────────────────
    tram_profile = profile.get("tram_routes", {})
    if tram_profile.get("method") == "colour":
        print("    Extracting tram routes...")
        tram_mask = extract_features_by_colour(
            img,
            tram_profile["hsv_lower"],
            tram_profile["hsv_upper"]
        )
        tram_skeleton = skeletonize_roads(tram_mask)
        tram_gdf = raster_to_lines(tram_skeleton, transform)
        tram_gdf["year"]   = map_year
        tram_gdf["type"]   = "tram"
        tram_gdf["source"] = Path(map_path).stem

        tram_output = Path(output_dir) / f"{stem}_trams.gpkg"
        if not tram_gdf.empty:
            tram_gdf.to_file(tram_output, driver="GPKG")
            print(f"    Trams: {len(tram_gdf)} features → {tram_output.name}")
        results["trams"] = str(tram_output)

    return results


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 60)
    print("CALCUTTA HISTORICAL MAPS — FEATURE EXTRACTION")
    print("=" * 60)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Get all georeferenced maps
    georef_maps = list(Path(INPUT_DIR).glob("*_georef.tif"))
    print(f"\nFound {len(georef_maps)} georeferenced maps")

    if not georef_maps:
        print("\nNO GEOREFERENCED MAPS FOUND")
        print("Complete Phase 2 (georeferencing) first")
        print(f"Expected location: {INPUT_DIR}")
        return

    all_results = {}
    for map_path in tqdm(georef_maps, desc="Extracting features"):
        # Determine profile from filename
        stem = map_path.stem
        profile_name = "pre_1870"  # default
        year = 0

        for key, profile in MAP_PROFILES.items():
            if key in stem:
                profile_name = profile
                year = int(key.split("_")[1]) if "_" in key else 0
                break

        results = extract_all_features(
            map_path, profile_name, OUTPUT_DIR, year
        )
        all_results[stem] = results

    # Save extraction report
    report_path = Path(OUTPUT_DIR) / "extraction_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"Vectors saved to: {OUTPUT_DIR}")
    print(f"Report: {report_path}")
    print(f"{'='*60}")
    print("\nNEXT STEP: Open vectors in QGIS for manual cleaning")
    print("Run: 04_validate.py to check topology")


if __name__ == "__main__":
    main()

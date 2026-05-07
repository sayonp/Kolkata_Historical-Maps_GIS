"""
PHASE 1: MAP STANDARDISATION
==============================
Converts all historical Calcutta maps to:
- Standard format: GeoTIFF
- Standard resolution: 300 DPI
- Standard colour space: RGB
- Standard bit depth: 8-bit
- Consistent naming convention

Requirements:
    pip install Pillow numpy rasterio gdal opencv-python tqdm
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
import json
import csv
from datetime import datetime

# ─────────────────────────────────────────
# CONFIGURATION — Edit these paths
# ─────────────────────────────────────────
INPUT_DIR  = "/Users/xon/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Maps/Sourced Maps"
OUTPUT_DIR = "/Users/xon/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Maps/Processed"
LOG_FILE   = "/Users/xon/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Maps/processing_log.csv"

TARGET_DPI        = 300
TARGET_FORMAT     = "TIFF"
TARGET_MODE       = "RGB"
TARGET_MAX_SIZE   = 20000  # max pixels on longest side

# ─────────────────────────────────────────
# MAP METADATA — Add details for each map
# ─────────────────────────────────────────
MAP_METADATA = {
    "Calcutta_1735-84":  {"year": 1735, "source": "Unknown",         "scale": "Unknown", "type": "city"},
    "Calcutta_1756":     {"year": 1756, "source": "Unknown",         "scale": "Unknown", "type": "city"},
    "Calcutta_1756-57":  {"year": 1756, "source": "Unknown",         "scale": "Unknown", "type": "city"},
    "Calcutta_1832":     {"year": 1832, "source": "Unknown",         "scale": "Unknown", "type": "city"},
    "Calcutta_1842":     {"year": 1842, "source": "SDUK",            "scale": "Unknown", "type": "city"},
    "Calcutta_1842_Fullsheet": {"year": 1842, "source": "SDUK",     "scale": "Unknown", "type": "city"},
    "Calcutta_1847-49_survey": {"year": 1847, "source": "LOC/Simms","scale": "1:31680", "type": "survey"},
    "Calcutta_1847-1849_DR":   {"year": 1847, "source": "David Rumsey","scale": "1:31680", "type": "survey"},
    "Calcutta_1857":     {"year": 1857, "source": "LOC",             "scale": "1:31680", "type": "survey"},
    "Calcutta_1857_environ": {"year": 1857, "source": "Unknown",     "scale": "Unknown", "type": "regional"},
    "Calcutta_1865":     {"year": 1865, "source": "Johnson",         "scale": "Unknown", "type": "city"},
    "Calcutta_1893":     {"year": 1893, "source": "Constable",       "scale": "Unknown", "type": "city"},
    "Calcutta_1894":     {"year": 1894, "source": "Unknown",         "scale": "Unknown", "type": "city"},
    "Calcutta_1911":     {"year": 1911, "source": "IGI",             "scale": "Unknown", "type": "city"},
    "Calcutta_1912":     {"year": 1912, "source": "Unknown",         "scale": "Unknown", "type": "city"},
    "Calcutta_1914":     {"year": 1914, "source": "Unknown",         "scale": "Unknown", "type": "city"},
    "Calcutta_1924":     {"year": 1924, "source": "Unknown",         "scale": "Unknown", "type": "city"},
    "Calcutta_Environs_1924": {"year": 1924, "source": "Unknown",    "scale": "Unknown", "type": "regional"},
    "Hooghly_River_1749": {"year": 1749, "source": "Unknown",        "scale": "Unknown", "type": "river"},
}

# ─────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────

def get_all_maps(input_dir):
    """Find all image files in input directory"""
    extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    maps = []
    for f in Path(input_dir).iterdir():
        if f.suffix.lower() in extensions:
            maps.append(f)
    # Also handle ZIP files (like Calcutta_1847-1849_DR.zip)
    return sorted(maps)


def analyse_image(img_path):
    """Analyse image properties before processing"""
    try:
        with Image.open(img_path) as img:
            info = {
                "filename": img_path.name,
                "format":   img.format,
                "mode":     img.mode,
                "width":    img.width,
                "height":   img.height,
                "dpi":      img.info.get("dpi", "Unknown"),
            }
        return info
    except Exception as e:
        return {"filename": img_path.name, "error": str(e)}


def deskew_image(img):
    """
    Auto-deskew a scanned map image using Hough line detection.
    Returns corrected image.
    """
    try:
        import cv2
        img_cv = np.array(img.convert("L"))  # grayscale
        edges  = cv2.Canny(img_cv, 50, 150, apertureSize=3)
        lines  = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is None:
            return img  # no lines found, return original

        angles = []
        for line in lines[:20]:  # use top 20 lines only
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            if abs(angle) < 10:  # only small corrections
                angles.append(angle)

        if not angles:
            return img

        median_angle = np.median(angles)
        if abs(median_angle) > 0.5:  # only correct if > 0.5 degrees
            img = img.rotate(-median_angle, expand=True,
                             fillcolor=(255, 255, 255))
            print(f"    Deskewed by {median_angle:.2f} degrees")

        return img
    except ImportError:
        print("    OpenCV not available — skipping deskew")
        return img


def enhance_image(img):
    """
    Enhance historical map for better feature extraction.
    Improves contrast and sharpness.
    """
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)

    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.5)

    return img


def resize_if_needed(img, max_size=TARGET_MAX_SIZE):
    """Resize image if it exceeds maximum size"""
    w, h = img.size
    if max(w, h) > max_size:
        ratio  = max_size / max(w, h)
        new_w  = int(w * ratio)
        new_h  = int(h * ratio)
        img    = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"    Resized from {w}x{h} to {new_w}x{new_h}")
    return img


def standardise_map(input_path, output_dir, enhance=True, deskew=True):
    """
    Main standardisation function for a single map.
    Returns dict with processing results.
    """
    result = {
        "input":     str(input_path),
        "status":    "pending",
        "output":    None,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        print(f"\n  Processing: {input_path.name}")

        # Open image
        with Image.open(input_path) as img:
            print(f"    Original: {img.mode}, {img.width}x{img.height}, {img.format}")

            # Convert to RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
                print(f"    Converted to RGB")

            # Deskew
            if deskew:
                img = deskew_image(img)

            # Enhance
            if enhance:
                img = enhance_image(img)

            # Resize if too large
            img = resize_if_needed(img)

            # Build output filename
            stem        = input_path.stem
            output_name = f"{stem}_processed.tif"
            output_path = Path(output_dir) / output_name

            # Save as TIFF with DPI metadata
            img.save(
                output_path,
                format="TIFF",
                dpi=(TARGET_DPI, TARGET_DPI),
                compression="lzw"
            )

            result["status"] = "success"
            result["output"] = str(output_path)
            result["width"]  = img.width
            result["height"] = img.height
            print(f"    Saved: {output_name}")

    except Exception as e:
        result["status"] = "error"
        result["error"]  = str(e)
        print(f"    ERROR: {e}")

    return result


def generate_report(results, output_dir):
    """Generate processing report as CSV"""
    report_path = Path(output_dir) / "processing_report.csv"

    fieldnames = ["input", "output", "status", "width",
                  "height", "error", "timestamp"]

    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row)

    print(f"\nReport saved: {report_path}")
    return report_path


def generate_metadata_json(output_dir):
    """Save metadata JSON alongside processed maps"""
    meta_path = Path(output_dir) / "map_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(MAP_METADATA, f, indent=2)
    print(f"Metadata saved: {meta_path}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 60)
    print("CALCUTTA HISTORICAL MAPS — STANDARDISATION PIPELINE")
    print("=" * 60)

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Analyse all maps first
    print("\n[1/3] Analysing input maps...")
    maps = get_all_maps(INPUT_DIR)
    print(f"  Found {len(maps)} map files")

    for m in maps:
        info = analyse_image(m)
        print(f"  {info.get('filename')}: "
              f"{info.get('width')}x{info.get('height')} "
              f"{info.get('mode')} {info.get('dpi')} DPI")

    # Process all maps
    print(f"\n[2/3] Standardising maps to {TARGET_DPI}DPI RGB TIFF...")
    results = []
    for map_path in tqdm(maps, desc="Processing"):
        result = standardise_map(map_path, OUTPUT_DIR)
        results.append(result)

    # Generate report
    print("\n[3/3] Generating report...")
    generate_report(results, OUTPUT_DIR)
    generate_metadata_json(OUTPUT_DIR)

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    errors  = sum(1 for r in results if r["status"] == "error")
    print(f"\n{'='*60}")
    print(f"COMPLETE: {success} processed, {errors} errors")
    print(f"Output:   {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

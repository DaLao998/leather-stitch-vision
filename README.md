# Zuodian Pattern Analysis

Lightweight image-processing pipeline for perforated seat or fabric patterns.

It takes raw images, crops the target ROI, extracts hole centers, rebuilds pattern regions, and can further analyze centerlines and compare geometry across samples.

## What It Does

- Crop a polygon ROI from a full image
- Extract perforation holes as a binary mask
- Compute hole centers
- Rebuild pattern instance regions from hole layout
- Detect centerlines and main geometric segments
- Compare geometric structure between samples

## Pipeline

```text
raw image
  -> ROI crop
  -> hole extraction
  -> pattern region reconstruction
  -> preview / binary outputs
  -> centerline + geometry analysis
  -> cross-sample comparison
```

## Project Structure

```text
src/
  main.py
  main1.py
  crop_seat_roi.py
  extract_holes.py
  find_boundary.py
  utils.py
  patterns/
    centerline.py
    compare_geo_segments.py
utils/
  roi_select.py
picture/
output/
md/
```

## Files

### Core scripts

- [`src/main.py`](/E:/MU/priority/zuodian/src/main.py): Main end-to-end pipeline for ROI crop, hole extraction, and pattern reconstruction.
- [`src/main1.py`](/E:/MU/priority/zuodian/src/main1.py): Alternate entry script with a different default parameter set.
- [`src/crop_seat_roi.py`](/E:/MU/priority/zuodian/src/crop_seat_roi.py): Polygon ROI cropping without perspective warp.
- [`src/extract_holes.py`](/E:/MU/priority/zuodian/src/extract_holes.py): Hole mask extraction and hole-center calculation.
- [`src/find_boundary.py`](/E:/MU/priority/zuodian/src/find_boundary.py): Pattern instance / boundary reconstruction from hole centers.
- [`src/utils.py`](/E:/MU/priority/zuodian/src/utils.py): Shared helpers for image IO, path iteration, and output handling.

### Pattern analysis

- [`src/patterns/centerline.py`](/E:/MU/priority/zuodian/src/patterns/centerline.py): Centerline extraction, line fitting, node detection, and geometry JSON export.
- [`src/patterns/compare_geo_segments.py`](/E:/MU/priority/zuodian/src/patterns/compare_geo_segments.py): Matches nodes and segments between geometry JSON files and visualizes the result.

### Utilities

- [`utils/roi_select.py`](/E:/MU/priority/zuodian/utils/roi_select.py): Manual point-picking tool for ROI corner selection.

## Install

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Quick Start

Run the main pipeline:

```bash
python src/main.py --input picture/2.jpg --output output/pattern
```

Run ROI crop only:

```bash
python src/crop_seat_roi.py --input picture/3.jpg --output output/roi
```

Run hole extraction only:

```bash
python src/extract_holes.py --input output/roi/3_crop.jpg --output output/holes --save-preview
```

Run centerline / geometry analysis:

```bash
python src/patterns/centerline.py --input output/pattern
```

Compare geometry results:

```bash
python src/patterns/compare_geo_segments.py --ref output/centerline/geo_1.json --targets output/centerline/geo_2.json output/centerline/geo_3.json
```

## Typical Outputs

- `*_crop.jpg`: cropped ROI
- `*_holes_bw.png`: hole mask
- `*_pattern_preview.png`: overlay preview
- `*_matrix_instances_bw.png`: reconstructed pattern instances
- `geo_*.jpg`: geometry visualization
- `geo_*.json`: structured geometry result
- `*_vs_*.json`: comparison result
- `*_vs_*_vis.png`: comparison visualization

## Notes

- `opencv-contrib-python` is required because the project uses `cv2.ximgproc.thinning`.
- `scipy` is required for nearest-neighbor search in boundary reconstruction.
- More detailed algorithm notes are in [`md/Algorithm Process.md`](/E:/MU/priority/zuodian/md/Algorithm%20Process.md).

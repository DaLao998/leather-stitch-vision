from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from src.utils import (
    build_hole_response,
    iter_image_paths,
    load_bgr_image,
    longest_active_span,
    smooth_projection,
    write_image,
)


def extract_hole_centers(image: np.ndarray) -> np.ndarray:
    response = build_hole_response(image, kernel_divisor=100, min_kernel=7)
    _, binary = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    centers = np.zeros_like(binary)

    for label in range(1, num_labels):
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 2 or area > 120:
            continue

        short_side = max(1, min(w, h))
        aspect_ratio = max(w, h) / float(short_side)
        fill_ratio = area / float(max(1, w * h))
        if aspect_ratio > 1.9 or fill_ratio < 0.18:
            continue

        cx, cy = centroids[label]
        cv2.circle(centers, (int(round(cx)), int(round(cy))), 1, 255, -1)

    return centers


def find_cushion_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    original_h, original_w = image.shape[:2]
    scale = min(1.0, 1200.0 / float(max(original_h, original_w)))
    resized = (
        cv2.resize(
            image,
            (int(round(original_w * scale)), int(round(original_h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1.0
        else image.copy()
    )

    centers = extract_hole_centers(resized)
    if cv2.countNonZero(centers) < 200:
        raise RuntimeError("Failed to locate enough perforation holes for ROI detection.")

    h, w = centers.shape
    col_projection = smooth_projection(centers.sum(axis=0), window=max(15, w // 40))
    row_projection = smooth_projection(centers.sum(axis=1), window=max(15, h // 40))

    col_mask = col_projection > (float(col_projection.max()) * 0.30)
    row_mask = row_projection > (float(row_projection.max()) * 0.30)
    x0_small, x1_small = longest_active_span(col_mask)
    y0_small, y1_small = longest_active_span(row_mask)

    if x1_small <= x0_small or y1_small <= y0_small:
        raise RuntimeError("Failed to derive a valid ROI from perforation density.")

    pad_x_small = max(8, int(round((x1_small - x0_small + 1) * 0.07)))
    pad_y_small = max(10, int(round((y1_small - y0_small + 1) * 0.09)))

    x0_small = max(0, x0_small - pad_x_small)
    y0_small = max(0, y0_small - pad_y_small)
    x1_small = min(w - 1, x1_small + pad_x_small)
    y1_small = min(h - 1, y1_small + pad_y_small)

    inv_scale = 1.0 / scale
    x0 = max(0, int(np.floor(x0_small * inv_scale)))
    y0 = max(0, int(np.floor(y0_small * inv_scale)))
    x1 = min(original_w, int(np.ceil((x1_small + 1) * inv_scale)))
    y1 = min(original_h, int(np.ceil((y1_small + 1) * inv_scale)))
    return x0, y0, x1, y1


def crop_and_draw(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = find_cushion_bbox(image)
    crop = image[y0:y1, x0:x1].copy()
    preview = image.copy()
    cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 0, 255), 3)
    return crop, preview


def process_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return crop_and_draw(image)


def process_path(image_path: Path, output_dir: Path) -> None:
    image = load_bgr_image(image_path)
    crop, preview = process_image(image)

    stem = image_path.stem
    write_image(output_dir / f"{stem}_crop.jpg", crop)
    write_image(output_dir / f"{stem}_preview.jpg", preview)


def main() -> None:
    start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Crop the perforated cushion ROI from sample images.")
    parser.add_argument("--input", default="picture/1.jpg", help="Image file or directory.")
    parser.add_argument("--output", default="output/roi", help="Directory for result images.")
    args = parser.parse_args()

    image_paths = iter_image_paths(Path(args.input))
    if not image_paths:
        raise RuntimeError(f"No images found under: {args.input}")

    output_dir = Path(args.output)
    for image_path in image_paths:
        process_path(image_path, output_dir)

    print(f"roi: {time.perf_counter() - start:.3f}s")


if __name__ == "__main__":
    main()

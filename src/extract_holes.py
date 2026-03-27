from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from src.utils import iter_image_paths, load_bgr_image, longest_active_span, smooth_projection, write_image


CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
BLACKHAT_KERNEL_SIZE = 7
MIN_FILL_RATIO = 0.35


def build_single_channel_response(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    enhanced = clahe.apply(gray)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (BLACKHAT_KERNEL_SIZE, BLACKHAT_KERNEL_SIZE),
    )
    response = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
    return enhanced, response


def build_hole_artifacts(
    image: np.ndarray,
    min_area: int = 6,
    max_area: int = 120,
    max_aspect_ratio: float = 1.8,
    response: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, response = build_single_channel_response(image) if response is None else (None, response)
    _, binary = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    border_labels = np.unique(
        np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
    )
    border_mask = np.zeros(num_labels, dtype=bool)
    border_mask[border_labels] = True
    keep = np.zeros(num_labels, dtype=bool)

    for label in range(1, num_labels):
        if border_mask[label]:
            continue

        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        aspect_ratio = max(width, height) / max(1, min(width, height))
        fill_ratio = area / max(1, width * height)

        if (
            min_area <= area <= max_area
            and aspect_ratio <= max_aspect_ratio
            and fill_ratio >= MIN_FILL_RATIO
        ):
            keep[label] = True

    hole_mask = np.where(keep[labels], 255, 0).astype(np.uint8)
    hole_mask = constrain_to_dense_hole_bbox(hole_mask)
    inverted = cv2.bitwise_not(hole_mask)
    preview = image.copy()
    preview[hole_mask > 0] = (255, 255, 255)

    return response, hole_mask, inverted, preview


def constrain_to_dense_hole_bbox(hole_mask: np.ndarray) -> np.ndarray:
    binary = (hole_mask > 0).astype(np.uint8)
    if int(binary.sum()) == 0:
        return hole_mask

    h, w = binary.shape
    col_projection = smooth_projection(binary.sum(axis=0), window=max(15, w // 45))
    row_projection = smooth_projection(binary.sum(axis=1), window=max(15, h // 45))

    col_mask = col_projection > (float(col_projection.max()) * 0.22)
    row_mask = row_projection > (float(row_projection.max()) * 0.22)
    x0, x1 = longest_active_span(col_mask)
    y0, y1 = longest_active_span(row_mask)

    if x1 <= x0 or y1 <= y0:
        return hole_mask

    pad_x = max(6, int(round((x1 - x0 + 1) * 0.02)))
    pad_y = max(6, int(round((y1 - y0 + 1) * 0.02)))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w - 1, x1 + pad_x)
    y1 = min(h - 1, y1 + pad_y)

    constrained = np.zeros_like(hole_mask)
    constrained[y0 : y1 + 1, x0 : x1 + 1] = hole_mask[y0 : y1 + 1, x0 : x1 + 1]
    return constrained


def process_path(
    image_path: Path,
    output_dir: Path,
    min_area: int,
    max_area: int,
    max_aspect_ratio: float,
) -> None:
    image = load_bgr_image(image_path)
    response, hole_mask, inverted, preview = build_hole_artifacts(
        image,
        min_area=min_area,
        max_area=max_area,
        max_aspect_ratio=max_aspect_ratio,
    )

    stem = image_path.stem
    write_image(output_dir / f"{stem}_blackhat.png", response)
    write_image(output_dir / f"{stem}_holes_bw.png", hole_mask)
    write_image(output_dir / f"{stem}_holes_bw_inverted.png", inverted)
    write_image(output_dir / f"{stem}_holes_preview.png", preview)


def main() -> None:
    start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Extract perforation holes into a binary image.")
    parser.add_argument("--input", default="output/roi/3_crop.jpg", help="Image file or directory.")
    parser.add_argument("--output", default="output/holes", help="Directory for result images.")
    parser.add_argument("--min-area", type=int, default=6, help="Minimum connected-component area.")
    parser.add_argument("--max-area", type=int, default=120, help="Maximum connected-component area.")
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=1.8,
        help="Reject components more elongated than this ratio.",
    )
    args = parser.parse_args()

    image_paths = iter_image_paths(Path(args.input))
    if not image_paths:
        raise RuntimeError(f"No images found under: {args.input}")

    output_dir = Path(args.output)
    for image_path in image_paths:
        process_path(
            image_path=image_path,
            output_dir=output_dir,
            min_area=args.min_area,
            max_area=args.max_area,
            max_aspect_ratio=args.max_aspect_ratio,
        )

    print(f"holes: {time.perf_counter() - start:.3f}s")


if __name__ == "__main__":
    main()

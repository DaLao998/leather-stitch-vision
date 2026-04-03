from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from src.crop_seat_roi import build_roi_crop
from src.extract_holes import RESPONSE_MODES, build_hole_mask_and_centers
from src.utils import iter_image_paths, load_bgr_image, write_image


def build_preview_with_method(
    boundary_method: int,
    centers,
    crop,
    hole_mask_shape,
):
    if boundary_method == 1:
        from src.find_boundary_hole1 import build_pattern_preview_from_centers

        return build_pattern_preview_from_centers(
            centers=centers,
            roi_image=crop,
            mask_shape=hole_mask_shape,
            copy_image=False,
        )

    if boundary_method == 2:
        from src.find_boundary_hole2 import build_boundary_line_mask_from_centers

        line_mask = build_boundary_line_mask_from_centers(
            shape=hole_mask_shape,
            centers=centers,
        )
        line_mask = cv2.dilate(
            line_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

        preview = crop
        preview[line_mask > 0] = (0, 255, 0)
        return preview

    raise ValueError(f"Unsupported boundary method: {boundary_method}")


def process_path(
    image_path: Path,
    output_dir: Path,
    min_area: int,
    max_area: int,
    max_aspect_ratio: float,
    response_mode: str,
    boundary_method: int,
) -> tuple[float, float, float]:
    image = load_bgr_image(image_path)

    t1_start = time.perf_counter()
    crop = build_roi_crop(image)
    elapsed1 = time.perf_counter() - t1_start

    t2_start = time.perf_counter()
    _, hole_mask, centers = build_hole_mask_and_centers(
        crop,
        min_area=min_area,
        max_area=max_area,
        max_aspect_ratio=max_aspect_ratio,
        response_mode=response_mode,
    )
    elapsed2 = time.perf_counter() - t2_start

    t3_start = time.perf_counter()
    pattern_preview = build_preview_with_method(
        boundary_method=boundary_method,
        centers=centers,
        crop=crop,
        hole_mask_shape=hole_mask.shape,
    )
    elapsed3 = time.perf_counter() - t3_start

    stem = image_path.stem
    write_image(output_dir / f"{stem}_pattern_preview.png", pattern_preview)

    return elapsed1, elapsed2, elapsed3


def main() -> None:
    total_start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Run ROI crop, hole extraction, and boundary preview generation."
    )
    parser.add_argument("--input", default="picture/1.jpg", help="Image file or directory.")
    parser.add_argument("--output", default="output/pattern", help="Directory for final preview images.")
    parser.add_argument("--min-area", type=int, default=6, help="Minimum connected-component area for hole extraction.")
    parser.add_argument("--max-area", type=int, default=120, help="Maximum connected-component area for hole extraction.")
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=1.8,
        help="Reject hole candidates more elongated than this ratio.",
    )
    parser.add_argument(
        "--response-mode",
        choices=RESPONSE_MODES,
        default="multi",
        help="Hole response mode: multi is more robust, gray is faster.",
    )
    parser.add_argument(
        "--boundary-method",
        type=int,
        choices=(1, 2),
        default=2,
        help="Boundary method: 1 outputs boundary points preview, 2 outputs boundary line preview.",
    )
    args = parser.parse_args()

    image_paths = iter_image_paths(Path(args.input))
    if not image_paths:
        raise RuntimeError(f"No images found under: {args.input}")

    output_dir = Path(args.output)
    total1 = 0.0
    total2 = 0.0
    total3 = 0.0

    for image_path in image_paths:
        elapsed1, elapsed2, elapsed3 = process_path(
            image_path=image_path,
            output_dir=output_dir,
            min_area=args.min_area,
            max_area=args.max_area,
            max_aspect_ratio=args.max_aspect_ratio,
            response_mode=args.response_mode,
            boundary_method=args.boundary_method,
        )
        total1 += elapsed1
        total2 += elapsed2
        total3 += elapsed3

    total = total1 + total2 + total3
    total_time = time.perf_counter() - total_start
    print(
        f"耗时1: {total1:.3f}s + 耗时2: {total2:.3f}s + 耗时3: {total3:.3f}s = 纯算法总耗时: {total:.3f}s"
    )
    print(f"程序墙钟总耗时：{total_time:.3f} s")


if __name__ == "__main__":
    main()

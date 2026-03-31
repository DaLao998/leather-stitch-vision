from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.crop_seat_roi import build_roi_crop
from src.extract_holes import build_hole_artifacts
from src.find_boundary_hole_centerlines import build_pattern_preview
from src.utils import iter_image_paths, load_bgr_image, write_image


def process_path(
    image_path: Path,
    output_dir: Path,
    min_area: int,
    max_area: int,
    max_aspect_ratio: float,
) -> tuple[float, float, float]:
    image = load_bgr_image(image_path)

    crop, stage1_elapsed = build_roi_crop(image)

    stage2_start = time.perf_counter()
    _, hole_mask, _, _ = build_hole_artifacts(
        crop,
        min_area=min_area,
        max_area=max_area,
        max_aspect_ratio=max_aspect_ratio,
        build_inverted=False,
        build_preview=False,
    )
    stage2_elapsed = time.perf_counter() - stage2_start

    stage3_start = time.perf_counter()
    pattern_preview = build_pattern_preview(hole_mask, crop)
    stage3_elapsed = time.perf_counter() - stage3_start

    stem = image_path.stem
    write_image(output_dir / f"{stem}_pattern_preview.png", pattern_preview)
    return stage1_elapsed, stage2_elapsed, stage3_elapsed


def main() -> None:
    wall_start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Run ROI crop, hole extraction, and boundary-hole preview generation.")
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
    args = parser.parse_args()

    image_paths = iter_image_paths(Path(args.input))
    if not image_paths:
        raise RuntimeError(f"No images found under: {args.input}")

    output_dir = Path(args.output)
    stage1_total = 0.0
    stage2_total = 0.0
    stage3_total = 0.0

    for image_path in image_paths:
        stage1_elapsed, stage2_elapsed, stage3_elapsed = process_path(
            image_path=image_path,
            output_dir=output_dir,
            min_area=args.min_area,
            max_area=args.max_area,
            max_aspect_ratio=args.max_aspect_ratio,
        )
        stage1_total += stage1_elapsed
        stage2_total += stage2_elapsed
        stage3_total += stage3_elapsed

    total_compute = stage1_total + stage2_total + stage3_total
    wall_elapsed = time.perf_counter() - wall_start
    print(
        f"roi_compute: {stage1_total:.3f}s, "
        f"holes_compute: {stage2_total:.3f}s, "
        f"pattern_compute: {stage3_total:.3f}s, "
        f"total_compute: {total_compute:.3f}s, "
        f"wall: {wall_elapsed:.3f}s"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.crop_seat_roi import process_image as crop_roi
from src.extract_holes import build_hole_artifacts
from src.find_boundary_hole_centerlines import build_pattern_artifacts
from src.utils import iter_image_paths, load_bgr_image, write_image


def process_path(
    image_path: Path,
    output_root: Path,
    min_area: int,
    max_area: int,
    max_aspect_ratio: float,
    max_dim: int,
) -> None:
    image = load_bgr_image(image_path)
    crop, roi_preview = crop_roi(image)

    blackhat, hole_mask, inverted, hole_preview = build_hole_artifacts(
        crop,
        min_area=min_area,
        max_area=max_area,
        max_aspect_ratio=max_aspect_ratio,
    )
    band, skeleton, line_mask, pattern_preview = build_pattern_artifacts(hole_mask, crop, max_dim=max_dim)

    stem = image_path.stem
    write_image(output_root / "roi" / f"{stem}_crop.jpg", crop)
    write_image(output_root / "roi" / f"{stem}_preview.jpg", roi_preview)

    # write_image(output_root / "holes" / f"{stem}_blackhat.png", blackhat)
    write_image(output_root / "holes" / f"{stem}_holes_bw.png", hole_mask)
    # write_image(output_root / "holes" / f"{stem}_holes_bw_inverted.png", inverted)
    write_image(output_root / "holes" / f"{stem}_holes_preview.png", hole_preview)

    # write_image(output_root / "pattern" / f"{stem}_pattern_band.png", band)
    # write_image(output_root / "pattern" / f"{stem}_pattern_skeleton.png", skeleton)
    # write_image(output_root / "pattern" / f"{stem}_pattern_centerline.png", line_mask)
    write_image(output_root / "pattern" / f"{stem}_pattern_preview.png", pattern_preview)


def main() -> None:
    start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Run ROI crop, hole extraction, and pattern centerline detection together.")
    parser.add_argument("--input", default="picture/3.jpg", help="Image file or directory.")
    parser.add_argument("--output", default="output", help="Root directory for all stage outputs.")
    parser.add_argument("--min-area", type=int, default=6, help="Minimum connected-component area for hole extraction.")
    parser.add_argument("--max-area", type=int, default=120, help="Maximum connected-component area for hole extraction.")
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=1.8,
        help="Reject hole candidates more elongated than this ratio.",
    )
    parser.add_argument("--max-dim", type=int, default=1000, help="Processing size for pattern extraction.")
    args = parser.parse_args()

    image_paths = iter_image_paths(Path(args.input))
    if not image_paths:
        raise RuntimeError(f"No images found under: {args.input}")

    output_root = Path(args.output)
    for image_path in image_paths:
        process_path(
            image_path=image_path,
            output_root=output_root,
            min_area=args.min_area,
            max_area=args.max_area,
            max_aspect_ratio=args.max_aspect_ratio,
            max_dim=args.max_dim,
        )

    print(f"pipeline: {time.perf_counter() - start:.3f}s")


if __name__ == "__main__":
    main()

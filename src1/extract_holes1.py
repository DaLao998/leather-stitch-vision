from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from src.utils import build_hole_response, iter_image_paths, load_bgr_image, write_image


CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
BLACKHAT_KERNEL_SIZE = 7
MIN_FILL_RATIO = 0.35

RESPONSE_MODES = ("multi", "gray")


def build_single_channel_response(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    enhanced = clahe.apply(gray)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (BLACKHAT_KERNEL_SIZE, BLACKHAT_KERNEL_SIZE),
    )
    response = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
    return response


def build_response(
    image: np.ndarray,
    response_mode: str = "multi",
) -> np.ndarray:
    if response_mode == "multi":
        return build_hole_response(image, kernel_divisor=80, min_kernel=9)
    if response_mode == "gray":
        return build_single_channel_response(image)
    raise RuntimeError(f"Unsupported response_mode: {response_mode}")


def build_preview_image(image: np.ndarray, hole_mask: np.ndarray) -> np.ndarray:
    preview = image.copy()
    preview[hole_mask > 0] = 255
    return preview


def build_hole_mask_and_centers(
    image: np.ndarray,
    min_area: int = 6,
    max_area: int = 120,
    max_aspect_ratio: float = 1.8,
    response: np.ndarray | None = None,
    response_mode: str = "multi",
    discard_border_components: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if response is None:
        response = build_response(image, response_mode=response_mode)

    _, binary = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    areas = stats[:, cv2.CC_STAT_AREA].astype(np.int32)
    widths = stats[:, cv2.CC_STAT_WIDTH].astype(np.int32)
    heights = stats[:, cv2.CC_STAT_HEIGHT].astype(np.int32)

    short_side = np.maximum(1, np.minimum(widths, heights)).astype(np.float32)
    long_side = np.maximum(widths, heights).astype(np.float32)
    aspect_ratio = long_side / short_side
    fill_ratio = areas.astype(np.float32) / np.maximum(1, widths * heights).astype(np.float32)

    keep = (
        (areas >= int(min_area))
        & (areas <= int(max_area))
        & (aspect_ratio <= float(max_aspect_ratio))
        & (fill_ratio >= float(MIN_FILL_RATIO))
    )
    keep[0] = False

    if discard_border_components and num_labels > 1:
        border_ids = np.unique(
            np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
        )
        border_mask = np.zeros(num_labels, dtype=bool)
        border_mask[border_ids] = True
        keep &= ~border_mask
        keep[0] = False

    hole_mask = (keep[labels].astype(np.uint8) * 255)
    centers = centroids[keep].astype(np.float32)
    return response, hole_mask, centers


def build_hole_mask(
    image: np.ndarray,
    min_area: int = 6,
    max_area: int = 120,
    max_aspect_ratio: float = 1.8,
    response: np.ndarray | None = None,
    response_mode: str = "multi",
    discard_border_components: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    response, hole_mask, _ = build_hole_mask_and_centers(
        image=image,
        min_area=min_area,
        max_area=max_area,
        max_aspect_ratio=max_aspect_ratio,
        response=response,
        response_mode=response_mode,
        discard_border_components=discard_border_components,
    )
    return response, hole_mask


def build_hole_artifacts(
    image: np.ndarray,
    min_area: int = 6,
    max_area: int = 120,
    max_aspect_ratio: float = 1.8,
    response: np.ndarray | None = None,
    response_mode: str = "multi",
    discard_border_components: bool = False,
    build_inverted: bool = True,
    build_preview: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    response, hole_mask = build_hole_mask(
        image=image,
        min_area=min_area,
        max_area=max_area,
        max_aspect_ratio=max_aspect_ratio,
        response=response,
        response_mode=response_mode,
        discard_border_components=discard_border_components,
    )

    inverted = cv2.bitwise_not(hole_mask) if build_inverted else np.empty((0, 0), dtype=np.uint8)
    preview = build_preview_image(image, hole_mask) if build_preview else np.empty((0, 0, 3), dtype=np.uint8)
    return response, hole_mask, inverted, preview


def process_path(
    image_path: Path,
    output_dir: Path,
    min_area: int,
    max_area: int,
    max_aspect_ratio: float,
    response_mode: str = "multi",
    discard_border_components: bool = False,
    save_response: bool = False,
    save_inverted: bool = False,
    save_preview: bool = False,
) -> None:
    image = load_bgr_image(image_path)

    response, hole_mask, inverted, preview = build_hole_artifacts(
        image,
        min_area=min_area,
        max_area=max_area,
        max_aspect_ratio=max_aspect_ratio,
        response_mode=response_mode,
        discard_border_components=discard_border_components,
        build_inverted=save_inverted,
        build_preview=save_preview,
    )

    stem = image_path.stem
    write_image(output_dir / f"{stem}_holes_bw.png", hole_mask)

    if save_response:
        write_image(output_dir / f"{stem}_blackhat.png", response)
    if save_inverted:
        write_image(output_dir / f"{stem}_holes_bw_inverted.png", inverted)
    if save_preview:
        write_image(output_dir / f"{stem}_holes_preview.png", preview)


def main() -> None:
    wall_start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Extract perforation holes into a binary image.")
    parser.add_argument("--input", default="output/roi", help="Image file or directory.")
    parser.add_argument("--output", default="output/holes", help="Directory for result images.")
    parser.add_argument("--min-area", type=int, default=6, help="Minimum connected-component area.")
    parser.add_argument("--max-area", type=int, default=120, help="Maximum connected-component area.")
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=1.8,
        help="Reject components more elongated than this ratio.",
    )
    parser.add_argument(
        "--response-mode",
        choices=RESPONSE_MODES,
        default="multi",
        help="Hole response mode: multi is more robust to color changes, gray is faster.",
    )
    parser.add_argument(
        "--discard-border-components",
        action="store_true",
        help="Drop connected components touching the image border.",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save blackhat, inverted, and preview images.",
    )
    parser.add_argument(
        "--save-response",
        action="store_true",
        help="Save blackhat response image.",
    )
    parser.add_argument(
        "--save-inverted",
        action="store_true",
        help="Save inverted binary mask.",
    )
    parser.add_argument(
        "--save-preview",
        action="store_true",
        help="Save preview image with detected holes painted white.",
    )
    args = parser.parse_args()

    image_paths = iter_image_paths(Path(args.input), stem_suffix="_crop")
    if not image_paths:
        raise RuntimeError(f"No ROI crop images found under: {args.input}")

    save_response = args.save_debug or args.save_response
    save_inverted = args.save_debug or args.save_inverted
    save_preview = args.save_debug or args.save_preview

    output_dir = Path(args.output)
    for image_path in image_paths:
        process_path(
            image_path=image_path,
            output_dir=output_dir,
            min_area=args.min_area,
            max_area=args.max_area,
            max_aspect_ratio=args.max_aspect_ratio,
            response_mode=args.response_mode,
            discard_border_components=args.discard_border_components,
            save_response=save_response,
            save_inverted=save_inverted,
            save_preview=save_preview,
        )

    wall_elapsed = time.perf_counter() - wall_start
    print(f"wall: {wall_elapsed:.3f}s")


if __name__ == "__main__":
    main()

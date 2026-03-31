from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from src.utils import iter_image_paths, load_bgr_image, write_image


BASE_IMAGE_WIDTH = 5472
BASE_IMAGE_HEIGHT = 3648

ROI_POINTS = np.array(
    [
        [1689, 201],
        [3214, 228],
        [3180, 2445],
        [1584, 2382],
    ],
    dtype=np.float32,
)


def clip_points_to_image(points: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    h, w = image_shape[:2]
    pts = np.asarray(points, dtype=np.float32).copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts


def scale_points_to_image(
    points: np.ndarray,
    image_shape: tuple[int, ...],
    base_width: int = BASE_IMAGE_WIDTH,
    base_height: int = BASE_IMAGE_HEIGHT,
) -> np.ndarray:
    h, w = image_shape[:2]
    sx = w / float(base_width)
    sy = h / float(base_height)

    pts = np.asarray(points, dtype=np.float32).copy()
    pts[:, 0] *= sx
    pts[:, 1] *= sy
    return pts


def polygon_crop_without_warp(
    image: np.ndarray,
    points: np.ndarray,
    build_preview: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    algo_elapsed = 0.0

    t0 = time.perf_counter()
    pts = scale_points_to_image(points, image.shape)
    pts = clip_points_to_image(pts, image.shape)
    polygon = np.round(pts).astype(np.int32)
    x, y, w, h = cv2.boundingRect(polygon)
    if w <= 1 or h <= 1:
        raise RuntimeError(f"Invalid ROI bounding rect: x={x}, y={y}, w={w}, h={h}")
    algo_elapsed += time.perf_counter() - t0

    # 不计入 copy 时间
    crop = image[y:y + h, x:x + w].copy()

    t0 = time.perf_counter()
    local_polygon = polygon.copy()
    local_polygon[:, 0] -= x
    local_polygon[:, 1] -= y

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [local_polygon], 255)

    masked_crop = np.full_like(crop, 255)
    cv2.copyTo(crop, mask, masked_crop)
    algo_elapsed += time.perf_counter() - t0

    # 不计入 preview 整图 copy 时间
    preview = np.empty((0, 0, 3), dtype=np.uint8)
    if build_preview:
        preview = image.copy()

        t0 = time.perf_counter()
        cv2.polylines(preview, [polygon.reshape(-1, 1, 2)], True, (0, 0, 255), 3)

        for idx, (px, py) in enumerate(polygon, start=1):
            cv2.circle(preview, (px, py), 6, (0, 255, 255), -1)
            cv2.putText(
                preview,
                str(idx),
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        algo_elapsed += time.perf_counter() - t0

    return masked_crop, preview, algo_elapsed


def crop_and_draw(
    image: np.ndarray,
    points: np.ndarray = ROI_POINTS,
) -> tuple[np.ndarray, np.ndarray, float]:
    return polygon_crop_without_warp(image, points)


def build_roi_crop(
    image: np.ndarray,
    points: np.ndarray = ROI_POINTS,
) -> tuple[np.ndarray, float]:
    crop, _, algo_elapsed = polygon_crop_without_warp(image, points, build_preview=False)
    return crop, algo_elapsed


def process_path(
    image_path: Path,
    output_dir: Path,
    points: np.ndarray = ROI_POINTS,
) -> float:
    image = load_bgr_image(image_path)

    crop, preview, algo_elapsed = crop_and_draw(image, points=points)

    stem = image_path.stem

    # 写图仍然执行，但不计入 roi 时间
    write_image(output_dir / f"{stem}_crop.jpg", crop)
    write_image(output_dir / f"{stem}_preview.jpg", preview)

    return algo_elapsed


def parse_points_from_args(raw_points: list[float] | None) -> np.ndarray:
    if raw_points is None:
        return ROI_POINTS.copy()

    if len(raw_points) != 8:
        raise RuntimeError("--points must provide exactly 8 numbers: x1 y1 x2 y2 x3 y3 x4 y4")

    return np.asarray(raw_points, dtype=np.float32).reshape(4, 2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop a quadrilateral ROI without any geometric transform."
    )
    parser.add_argument("--input", default="picture/1.jpg", help="Image file or directory.")
    parser.add_argument("--output", default="output/roi", help="Directory for result images.")
    parser.add_argument(
        "--points",
        type=float,
        nargs=8,
        default=None,
        metavar=("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"),
        help="Four ROI corner points in image coordinates.",
    )

    args = parser.parse_args()
    points = parse_points_from_args(args.points)

    image_paths = iter_image_paths(Path(args.input))
    if not image_paths:
        raise RuntimeError(f"No images found under: {args.input}")

    output_dir = Path(args.output)

    roi_total = 0.0
    for image_path in image_paths:
        roi_total += process_path(image_path, output_dir, points=points)

    print(f"roi(exclude copy/write): {roi_total:.6f}s")


if __name__ == "__main__":
    main()

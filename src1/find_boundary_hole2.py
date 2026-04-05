from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from src1.utils import derive_roi_path, iter_mask_paths, load_binary_mask, load_bgr_image, write_image

try:
    from scipy.spatial import cKDTree
except ImportError as exc:
    raise RuntimeError("This script requires scipy. Please run: pip install scipy") from exc


def extract_hole_centers(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask if mask.dtype == np.uint8 else mask.astype(np.uint8)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), 8)

    h_img, w_img = mask.shape[:2]
    points: list[np.ndarray] = []

    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])

        if x <= 0 or y <= 0 or (x + w) >= w_img or (y + h) >= h_img:
            continue

        aspect = max(w / max(h, 1), h / max(w, 1))
        fill = area / max(w * h, 1)
        if 2 <= area <= 200 and aspect <= 3.0 and fill >= 0.2:
            points.append(centroids[idx])

    if len(points) < 16:
        raise RuntimeError("Too few valid hole centers.")

    return np.asarray(points, dtype=np.float32)


def estimate_spacing(points: np.ndarray, tree: cKDTree | None = None) -> float:
    if len(points) < 2:
        raise RuntimeError("Not enough points to estimate spacing.")

    if tree is None:
        tree = cKDTree(points)

    dists, _ = tree.query(points, k=2)
    nearest = dists[:, 1]
    nearest = nearest[np.isfinite(nearest)]
    if len(nearest) == 0:
        raise RuntimeError("Failed to estimate spacing.")
    return float(np.median(nearest))


def _max_angular_gap_deg(vectors: np.ndarray) -> float:
    if len(vectors) == 0:
        return 360.0

    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angles = np.sort(angles)
    wrapped = np.concatenate([angles, angles[:1] + 2.0 * np.pi])
    gaps = np.diff(wrapped)
    return float(np.max(np.degrees(gaps)))


def _count_occupied_sectors(vectors: np.ndarray, num_sectors: int = 8) -> int:
    if len(vectors) == 0:
        return 0

    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angles = (angles + 2.0 * np.pi) % (2.0 * np.pi)
    sector_ids = np.floor(angles / (2.0 * np.pi / num_sectors)).astype(np.int32)
    return int(len(np.unique(sector_ids)))


def extract_boundary_points_from_centers(
    centers: np.ndarray,
    spacing: float | None = None,
    tree: cKDTree | None = None,
) -> np.ndarray:
    if len(centers) < 16:
        raise RuntimeError("Too few valid hole centers.")

    if tree is None:
        tree = cKDTree(centers)
    if spacing is None:
        spacing = estimate_spacing(centers, tree=tree)

    min_r = spacing * 0.55
    max_r = spacing * 1.45
    min_r2 = min_r * min_r
    max_r2 = max_r * max_r

    boundary_idx: list[int] = []

    for idx, point in enumerate(centers):
        neighbor_ids = tree.query_ball_point(point, r=max_r)

        local_vectors: list[np.ndarray] = []
        px, py = float(point[0]), float(point[1])

        for nb in neighbor_ids:
            if nb == idx:
                continue

            dx = float(centers[nb, 0]) - px
            dy = float(centers[nb, 1]) - py
            dist2 = dx * dx + dy * dy

            if min_r2 <= dist2 <= max_r2:
                local_vectors.append(np.array([dx, dy], dtype=np.float32))

        if not local_vectors:
            boundary_idx.append(idx)
            continue

        local_vectors_np = np.asarray(local_vectors, dtype=np.float32)
        neighbor_count = int(len(local_vectors_np))
        occupied_sectors = _count_occupied_sectors(local_vectors_np, num_sectors=8)
        max_gap_deg = _max_angular_gap_deg(local_vectors_np)

        is_boundary = (
            neighbor_count <= 4
            or occupied_sectors <= 5
            or max_gap_deg >= 115.0
        )

        if is_boundary:
            boundary_idx.append(idx)

    if not boundary_idx:
        return np.empty((0, 2), dtype=np.float32)

    return centers[np.asarray(boundary_idx, dtype=np.int32)].astype(np.float32)


def extract_boundary_points(hole_mask: np.ndarray) -> np.ndarray:
    centers = extract_hole_centers(hole_mask)
    tree = cKDTree(centers)
    spacing = estimate_spacing(centers, tree=tree)
    return extract_boundary_points_from_centers(centers, spacing=spacing, tree=tree)


def build_boundary_mask_from_points(
    shape: tuple[int, int],
    boundary_points: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.round(boundary_points).astype(np.int32)
    for x, y in pts:
        cv2.circle(mask, (x, y), 1, 255, -1)
    return mask


def build_boundary_mask_from_centers(
    shape: tuple[int, int],
    centers: np.ndarray,
    boundary_points: np.ndarray | None = None,
    spacing: float | None = None,
    tree: cKDTree | None = None,
) -> np.ndarray:
    if boundary_points is None:
        boundary_points = extract_boundary_points_from_centers(centers, spacing=spacing, tree=tree)
    return build_boundary_mask_from_points(shape, boundary_points)


def build_boundary_mask(hole_mask: np.ndarray) -> np.ndarray:
    centers = extract_hole_centers(hole_mask)
    tree = cKDTree(centers)
    spacing = estimate_spacing(centers, tree=tree)
    boundary_points = extract_boundary_points_from_centers(centers, spacing=spacing, tree=tree)
    return build_boundary_mask_from_points(hole_mask.shape, boundary_points)


def build_boundary_line_mask_from_centers(
    shape: tuple[int, int],
    centers: np.ndarray,
    boundary_points: np.ndarray | None = None,
    spacing: float | None = None,
) -> np.ndarray:
    if spacing is None:
        center_tree = cKDTree(centers)
        spacing = estimate_spacing(centers, tree=center_tree)

    if boundary_points is None:
        center_tree = cKDTree(centers)
        boundary_points = extract_boundary_points_from_centers(centers, spacing=spacing, tree=center_tree)

    if len(boundary_points) == 0:
        return np.zeros(shape, dtype=np.uint8)

    boundary_tree = cKDTree(boundary_points)

    line_mask = np.zeros(shape, dtype=np.uint8)
    min_r = spacing * 0.70
    max_r = spacing * 1.55
    min_r2 = min_r * min_r
    max_r2 = max_r * max_r

    points_i32 = np.round(boundary_points).astype(np.int32)

    for i, point in enumerate(boundary_points):
        neighbor_ids = boundary_tree.query_ball_point(point, r=max_r)
        candidates: list[tuple[float, int]] = []

        px, py = float(point[0]), float(point[1])

        for nb in neighbor_ids:
            if nb == i:
                continue

            dx = float(boundary_points[nb, 0]) - px
            dy = float(boundary_points[nb, 1]) - py
            dist2 = dx * dx + dy * dy

            if min_r2 <= dist2 <= max_r2:
                candidates.append((dist2, nb))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0])
        p1 = tuple(points_i32[i])

        for _, nb in candidates[:3]:
            p2 = tuple(points_i32[nb])
            cv2.line(line_mask, p1, p2, 255, 1, lineType=cv2.LINE_AA)

    return line_mask


def build_boundary_line_mask(hole_mask: np.ndarray) -> np.ndarray:
    centers = extract_hole_centers(hole_mask)
    center_tree = cKDTree(centers)
    spacing = estimate_spacing(centers, tree=center_tree)
    boundary_points = extract_boundary_points_from_centers(centers, spacing=spacing, tree=center_tree)
    return build_boundary_line_mask_from_centers(
        hole_mask.shape,
        centers,
        boundary_points=boundary_points,
        spacing=spacing,
    )


def build_pattern_preview_from_centers(
    centers: np.ndarray,
    roi_image: np.ndarray,
    mask_shape: tuple[int, int],
    copy_image: bool = True,
    boundary_points: np.ndarray | None = None,
    spacing: float | None = None,
    tree: cKDTree | None = None,
) -> np.ndarray:
    if boundary_points is None:
        boundary_points = extract_boundary_points_from_centers(centers, spacing=spacing, tree=tree)

    boundary_mask = build_boundary_mask_from_points(mask_shape, boundary_points)

    points_thick = cv2.dilate(
        boundary_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    preview = roi_image.copy() if copy_image else roi_image
    preview[points_thick > 0] = (0, 255, 0)
    return preview


def build_pattern_preview(
    hole_mask: np.ndarray,
    roi_image: np.ndarray,
    copy_image: bool = True,
) -> np.ndarray:
    centers = extract_hole_centers(hole_mask)
    tree = cKDTree(centers)
    spacing = estimate_spacing(centers, tree=tree)
    boundary_points = extract_boundary_points_from_centers(centers, spacing=spacing, tree=tree)
    return build_pattern_preview_from_centers(
        centers,
        roi_image,
        hole_mask.shape,
        copy_image=copy_image,
        boundary_points=boundary_points,
        spacing=spacing,
        tree=tree,
    )


def process_path(hole_path: Path, roi_path: Path, output_dir: Path) -> None:
    hole_mask = load_binary_mask(hole_path)
    roi_image = load_bgr_image(roi_path)

    centers = extract_hole_centers(hole_mask)
    tree = cKDTree(centers)
    spacing = estimate_spacing(centers, tree=tree)
    boundary_points = extract_boundary_points_from_centers(centers, spacing=spacing, tree=tree)

    preview = build_pattern_preview_from_centers(
        centers,
        roi_image,
        hole_mask.shape,
        copy_image=True,
        boundary_points=boundary_points,
        spacing=spacing,
        tree=tree,
    )

    stem = hole_path.stem.replace("_holes_bw", "")
    write_image(output_dir / f"{stem}_pattern_preview.png", preview)


def main() -> None:
    wall_start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Find boundary holes and render a preview image.")
    parser.add_argument("--holes", default="output/holes", help="Hole mask file or directory.")
    parser.add_argument("--roi", default="output/roi", help="ROI image file or directory.")
    parser.add_argument("--output", default="output/pattern", help="Directory for pattern outputs.")
    args = parser.parse_args()

    hole_paths = iter_mask_paths(Path(args.holes))
    if not hole_paths:
        raise RuntimeError(f"No hole masks found in: {args.holes}")

    roi_input = Path(args.roi)
    output_dir = Path(args.output)
    for hole_path in hole_paths:
        roi_path = roi_input if roi_input.is_file() else derive_roi_path(hole_path, roi_input)
        process_path(hole_path, roi_path, output_dir)

    wall_elapsed = time.perf_counter() - wall_start
    print(f"wall: {wall_elapsed:.3f}s")


if __name__ == "__main__":
    main()
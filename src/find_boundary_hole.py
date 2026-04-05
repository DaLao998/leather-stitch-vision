from __future__ import annotations

import argparse
import time
from pathlib import Path
import cv2
import numpy as np

from src.utils import derive_roi_path, iter_mask_paths, load_binary_mask, load_bgr_image, write_image

try:
    from scipy.spatial import cKDTree
except ImportError as exc:
    raise RuntimeError("This script requires scipy. Please run: pip install scipy") from exc

def extract_hole_centers(
    mask: np.ndarray,
    center_min_area: int = 2,
    center_max_area: int = 200,
    center_max_aspect: float = 3.0,
    center_min_fill: float = 0.2,
) -> np.ndarray:
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
        if center_min_area <= area <= center_max_area and aspect <= center_max_aspect and fill >= center_min_fill:
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
    neighbor_count_max: int = 4,
    occupied_sectors_max: int = 5,
    max_gap_deg_min: float = 115.0,
) -> np.ndarray:
    if len(centers) < 16:
        raise RuntimeError("Too few valid hole centers.")

    if tree is None:
        tree = cKDTree(centers)
    if spacing is None:
        spacing = estimate_spacing(centers, tree=tree)

    min_r2 = (spacing * 0.55) ** 2
    max_r = spacing * 1.45
    max_r2 = max_r ** 2
    boundary_idx: list[int] = []

    for idx, point in enumerate(centers):
        neighbor_ids = tree.query_ball_point(point, r=max_r)
        local_vectors: list[np.ndarray] = []
        px, py = float(point[0]), float(point[1])

        for nb in neighbor_ids:
            if nb == idx:
                continue
            dx, dy = float(centers[nb, 0]) - px, float(centers[nb, 1]) - py
            dist2 = dx * dx + dy * dy
            if min_r2 <= dist2 <= max_r2:
                local_vectors.append(np.array([dx, dy], dtype=np.float32))

        if not local_vectors:
            boundary_idx.append(idx)
            continue

        local_vectors_np = np.asarray(local_vectors, dtype=np.float32)
        is_boundary = (
            len(local_vectors_np) <= neighbor_count_max
            or _count_occupied_sectors(local_vectors_np, num_sectors=8) <= occupied_sectors_max
            or _max_angular_gap_deg(local_vectors_np) >= max_gap_deg_min
        )

        if is_boundary:
            boundary_idx.append(idx)

    if not boundary_idx:
        return np.empty((0, 2), dtype=np.float32)
    return centers[np.asarray(boundary_idx, dtype=np.int32)].astype(np.float32)

def build_boundary_line_mask_from_centers(
    shape: tuple[int, int],
    centers: np.ndarray,
    boundary_points: np.ndarray | None = None,
    spacing: float | None = None,
    center_min_area: int = 2,
    center_max_area: int = 200,
    center_max_aspect: float = 3.0,
    center_min_fill: float = 0.2,
    neighbor_count_max: int = 4,
    occupied_sectors_max: int = 5,
    max_gap_deg_min: float = 115.0,
    line_min_r_ratio: float = 0.70,
    line_max_r_ratio: float = 1.55,
) -> np.ndarray:
    center_tree = cKDTree(centers)
    if spacing is None:
        spacing = estimate_spacing(centers, tree=center_tree)

    if boundary_points is None:
        boundary_points = extract_boundary_points_from_centers(
            centers, spacing=spacing, tree=center_tree,
            neighbor_count_max=neighbor_count_max,
            occupied_sectors_max=occupied_sectors_max,
            max_gap_deg_min=max_gap_deg_min,
        )

    line_mask = np.zeros(shape, dtype=np.uint8)
    if len(boundary_points) == 0:
        return line_mask

    boundary_tree = cKDTree(boundary_points)
    min_r2 = (spacing * line_min_r_ratio) ** 2
    max_r = spacing * line_max_r_ratio
    max_r2 = max_r ** 2
    points_i32 = np.round(boundary_points).astype(np.int32)

    for i, point in enumerate(boundary_points):
        neighbor_ids = boundary_tree.query_ball_point(point, r=max_r)
        candidates: list[tuple[float, int]] = []
        px, py = float(point[0]), float(point[1])

        for nb in neighbor_ids:
            if nb == i: continue
            dx, dy = float(boundary_points[nb, 0]) - px, float(boundary_points[nb, 1]) - py
            dist2 = dx * dx + dy * dy
            if min_r2 <= dist2 <= max_r2:
                candidates.append((dist2, nb))

        if not candidates: continue
        candidates.sort(key=lambda x: x[0])
        p1 = tuple(points_i32[i])

        for _, nb in candidates[:3]:
            p2 = tuple(points_i32[nb])
            cv2.line(line_mask, p1, p2, 255, 1, lineType=cv2.LINE_AA)

    return line_mask

def process_path(hole_path: Path, roi_path: Path, output_dir: Path, args: argparse.Namespace) -> None:
    hole_mask = load_binary_mask(hole_path)
    roi_image = load_bgr_image(roi_path)

    centers = extract_hole_centers(
        hole_mask,
        center_min_area=args.center_min_area,
        center_max_area=args.center_max_area,
        center_max_aspect=args.center_max_aspect,
        center_min_fill=args.center_min_fill,
    )
    
    line_mask = build_boundary_line_mask_from_centers(
        hole_mask.shape, 
        centers,
        neighbor_count_max=args.boundary_neighbor_max,
        occupied_sectors_max=args.boundary_sectors_max,
        max_gap_deg_min=args.boundary_gap_min,
        line_min_r_ratio=args.line_min_r,
        line_max_r_ratio=args.line_max_r,
    )
    
    line_mask = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    preview = roi_image.copy()
    preview[line_mask > 0] = (0, 255, 0)

    stem = hole_path.stem.replace("_holes_bw", "")
    write_image(output_dir / f"{stem}_pattern_line_preview.png", preview)

def main() -> None:
    wall_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Find boundary line holes and render a preview image.")
    # 默认单图寻边调试
    parser.add_argument("--holes", default="output/holes/3_crop_holes_bw.png", help="Single Hole mask file.")
    parser.add_argument("--roi", default="output/roi/3_crop.jpg", help="Corresponding single ROI image.")
    parser.add_argument("--output", default="output/pattern", help="Directory for pattern outputs.")

    parser.add_argument("--center-min-area", type=int, default=2)
    parser.add_argument("--center-max-area", type=int, default=200)
    parser.add_argument("--center-max-aspect", type=float, default=3.0)
    parser.add_argument("--center-min-fill", type=float, default=0.2)
    parser.add_argument("--boundary-neighbor-max", type=int, default=4)
    parser.add_argument("--boundary-sectors-max", type=int, default=5)
    parser.add_argument("--boundary-gap-min", type=float, default=115.0)
    parser.add_argument("--line-min-r", type=float, default=0.70)
    parser.add_argument("--line-max-r", type=float, default=1.55)

    args = parser.parse_args()

    hole_paths = iter_mask_paths(Path(args.holes))
    if not hole_paths:
        raise RuntimeError(f"No hole masks found in: {args.holes}")

    roi_input = Path(args.roi)
    output_dir = Path(args.output)
    for hole_path in hole_paths:
        roi_path = roi_input if roi_input.is_file() else derive_roi_path(hole_path, roi_input)
        process_path(hole_path, roi_path, output_dir, args)

    print(f"wall: {time.perf_counter() - wall_start:.3f}s")

if __name__ == "__main__":
    main()
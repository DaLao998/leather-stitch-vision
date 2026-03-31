from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.utils import derive_roi_path, iter_mask_paths, load_binary_mask, load_bgr_image, write_image

try:
    from scipy.spatial import Delaunay, cKDTree
except ImportError as exc:
    raise RuntimeError("This script requires scipy. Please run: pip install scipy") from exc


def extract_hole_centers(mask: np.ndarray) -> np.ndarray:
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)

    points: list[np.ndarray] = []
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])

        if x <= 0 or y <= 0 or (x + w) >= mask.shape[1] or (y + h) >= mask.shape[0]:
            continue

        aspect = max(w / max(h, 1), h / max(w, 1))
        fill = area / max(w * h, 1)
        if 2 <= area <= 200 and aspect <= 3.0 and fill >= 0.2:
            points.append(centroids[idx])

    if len(points) < 16:
        raise RuntimeError("Too few valid hole centers.")
    return np.asarray(points, dtype=np.float32)


def estimate_spacing(points: np.ndarray) -> float:
    if len(points) < 2:
        raise RuntimeError("Not enough points to estimate spacing.")

    dists, _ = cKDTree(points).query(points, k=2)
    nearest = dists[:, 1]
    nearest = nearest[np.isfinite(nearest)]
    if len(nearest) == 0:
        raise RuntimeError("Failed to estimate spacing.")
    return float(np.median(nearest))


def extract_mesh_boundaries(centers: np.ndarray, spacing: float) -> list[np.ndarray]:
    tri = Delaunay(centers)
    max_edge_len = spacing * 1.85

    valid_triangles: list[np.ndarray] = []
    for simplex in tri.simplices:
        p0, p1, p2 = centers[simplex]
        d01 = float(np.linalg.norm(p0 - p1))
        d12 = float(np.linalg.norm(p1 - p2))
        d20 = float(np.linalg.norm(p2 - p0))
        if d01 < max_edge_len and d12 < max_edge_len and d20 < max_edge_len:
            valid_triangles.append(simplex)

    edge_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    for triangle in valid_triangles:
        edges = (
            tuple(sorted((int(triangle[0]), int(triangle[1])))),
            tuple(sorted((int(triangle[1]), int(triangle[2])))),
            tuple(sorted((int(triangle[2]), int(triangle[0])))),
        )
        for edge in edges:
            edge_counts[edge] += 1

    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    adjacency: defaultdict[int, list[int]] = defaultdict(list)
    for u, v in boundary_edges:
        adjacency[u].append(v)
        adjacency[v].append(u)

    paths: list[np.ndarray] = []
    visited_edges: set[tuple[int, int]] = set()

    for u, v in boundary_edges:
        edge = tuple(sorted((u, v)))
        if edge in visited_edges:
            continue

        path = [u, v]
        visited_edges.add(edge)
        curr = v

        while True:
            next_node = None
            for neighbor in adjacency[curr]:
                candidate = tuple(sorted((curr, neighbor)))
                if candidate not in visited_edges:
                    next_node = neighbor
                    visited_edges.add(candidate)
                    break

            if next_node is None:
                break

            path.append(next_node)
            curr = next_node

        if len(path) > 2:
            paths.append(centers[path])

    return paths


def build_boundary_centers_mask(
    shape: tuple[int, int],
    boundary_paths: list[np.ndarray],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for points in boundary_paths:
        for point in points:
            cv2.circle(mask, (int(round(point[0])), int(round(point[1]))), 1, 255, -1)
    return mask


def build_pattern_preview(
    hole_mask: np.ndarray,
    roi_image: np.ndarray,
) -> np.ndarray:
    centers = extract_hole_centers(hole_mask)
    spacing = estimate_spacing(centers)
    boundary_paths = extract_mesh_boundaries(centers, spacing)
    boundary_mask = build_boundary_centers_mask(hole_mask.shape, boundary_paths)

    preview = roi_image.copy()
    points_thick = cv2.dilate(boundary_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    preview[points_thick > 0] = (0, 255, 0)
    return preview


def process_path(hole_path: Path, roi_path: Path, output_dir: Path) -> float:
    hole_mask = load_binary_mask(hole_path)
    roi_image = load_bgr_image(roi_path)

    compute_start = time.perf_counter()
    preview = build_pattern_preview(hole_mask, roi_image)
    compute_elapsed = time.perf_counter() - compute_start

    stem = hole_path.stem.replace("_holes_bw", "")
    write_image(output_dir / f"{stem}_pattern_preview.png", preview)
    return compute_elapsed


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
    compute_total = 0.0

    for hole_path in hole_paths:
        roi_path = roi_input if roi_input.is_file() else derive_roi_path(hole_path, roi_input)
        compute_total += process_path(hole_path, roi_path, output_dir)

    wall_elapsed = time.perf_counter() - wall_start
    print(f"pattern_compute: {compute_total:.3f}s, wall: {wall_elapsed:.3f}s")


if __name__ == "__main__":
    main()

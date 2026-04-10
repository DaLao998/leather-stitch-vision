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


DEBUG_LAST: dict[str, object] = {
    "instance_mask": None,
    "boundary_mask": None,
    "signature_map": None,
    "boundary_hole_points": None,       # list[np.ndarray], 每个实例内被判定为边界孔洞的中心点
    "boundary_hole_points_mask": None,  # 所有边界孔洞点的汇总二值图
}


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return ivalue


def _nonnegative_float(value: str) -> float:
    fvalue = float(value)
    if fvalue < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return fvalue


def _positive_float(value: str) -> float:
    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return fvalue


def _ratio_0_1(value: str) -> float:
    fvalue = float(value)
    if not (0.0 <= fvalue <= 1.0):
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return fvalue


def validate_args(args: argparse.Namespace) -> None:
    if args.center_min_area > args.center_max_area:
        raise ValueError("center_min_area cannot be larger than center_max_area")
    if args.line_min_r >= args.line_max_r:
        raise ValueError("line_min_r must be smaller than line_max_r")
    if args.boundary_band_ratio > args.boundary_relax_ratio:
        raise ValueError("boundary_band_ratio cannot be larger than boundary_relax_ratio")


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

    if len(points) < 4:
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


def _build_structure_graph(
    centers: np.ndarray,
    spacing: float,
    tree: cKDTree,
    line_min_r_ratio: float,
    line_max_r_ratio: float,
) -> list[set[int]]:
    n = len(centers)
    graph = [set() for _ in range(n)]
    min_r = spacing * line_min_r_ratio
    max_r = spacing * line_max_r_ratio

    pairs = tree.query_pairs(r=max_r)
    for i, j in pairs:
        dx = float(centers[j, 0] - centers[i, 0])
        dy = float(centers[j, 1] - centers[i, 1])
        d = float(np.hypot(dx, dy))
        if d < min_r:
            continue
        graph[i].add(j)
        graph[j].add(i)
    return graph


def _connected_components_from_graph(graph: list[set[int]]) -> list[list[int]]:
    n = len(graph)
    visited = np.zeros((n,), dtype=bool)
    comps: list[list[int]] = []

    for i in range(n):
        if visited[i] or not graph[i]:
            continue
        stack = [i]
        visited[i] = True
        comp: list[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in graph[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def _max_angular_gap_deg(center: np.ndarray, neighbors: np.ndarray) -> float:
    if len(neighbors) < 2:
        return 360.0

    vec = neighbors - center[None, :]
    ang = np.arctan2(vec[:, 1], vec[:, 0])
    ang = np.sort((ang + 2.0 * np.pi) % (2.0 * np.pi))
    ang_wrap = np.concatenate([ang, ang[:1] + 2.0 * np.pi])
    gaps = np.diff(ang_wrap)
    return float(np.degrees(np.max(gaps)))


def _compute_boundary_holes_in_component(
    comp_points: np.ndarray,
    hull_for_dist: np.ndarray,
    spacing_fallback: float,
    line_min_r_ratio: float,
    line_max_r_ratio: float,
    boundary_band_ratio: float,
    boundary_relax_ratio: float,
    boundary_gap_deg: float,
) -> np.ndarray:
    n = len(comp_points)
    if n == 0:
        return np.zeros((0,), dtype=bool)
    if n <= 3:
        return np.ones((n,), dtype=bool)

    try:
        local_spacing = estimate_spacing(comp_points)
    except RuntimeError:
        local_spacing = spacing_fallback

    local_spacing = max(local_spacing, 1.0)
    min_r = local_spacing * line_min_r_ratio
    max_r = local_spacing * line_max_r_ratio

    tree = cKDTree(comp_points)
    flags = np.zeros((n,), dtype=bool)

    contour = hull_for_dist.astype(np.float32)

    for i in range(n):
        p = comp_points[i]

        # 到矩阵外边界（hull）的距离
        dist = abs(cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), True))

        # 只在当前矩阵内部取邻居
        idxs = tree.query_ball_point(p, r=max_r)
        idxs = [j for j in idxs if j != i]
        if idxs:
            neigh = comp_points[np.asarray(idxs, dtype=np.int32)]
            d = np.linalg.norm(neigh - p[None, :], axis=1)
            neigh = neigh[d >= min_r]
        else:
            neigh = np.empty((0, 2), dtype=np.float32)

        max_gap_deg = _max_angular_gap_deg(p, neigh)
        degree = len(neigh)

        hard_near_boundary = dist <= boundary_band_ratio * local_spacing
        soft_near_boundary = dist <= boundary_relax_ratio * local_spacing and max_gap_deg >= boundary_gap_deg
        sparse_boundary = degree <= 3 and dist <= 1.15 * local_spacing

        flags[i] = hard_near_boundary or soft_near_boundary or sparse_boundary

    return flags


def _draw_instance_region_mask_convex_hull(
    shape: tuple[int, int],
    centers: np.ndarray,
    components: list[list[int]],
    spacing: float,
    line_min_r_ratio: float,
    line_max_r_ratio: float,
    min_component_points: int,
    min_hull_area_ratio: float,
    smooth_eps_ratio: float,
    smooth_eps_min: float,
    hull_poly_eps_ratio: float,
    boundary_band_ratio: float,
    boundary_relax_ratio: float,
    boundary_gap_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    h, w = shape
    instance_mask = np.zeros((h, w), dtype=np.uint8)
    boundary_mask = np.zeros((h, w), dtype=np.uint8)
    signature_map = np.zeros((h, w), dtype=np.uint8)
    boundary_hole_points_mask = np.zeros((h, w), dtype=np.uint8)
    boundary_hole_points_list: list[np.ndarray] = []

    min_hull_area = max(12.0, (spacing * spacing) * min_hull_area_ratio)
    smooth_eps_base = max(smooth_eps_min, spacing * smooth_eps_ratio)

    for comp in components:
        if len(comp) < min_component_points:
            continue

        comp_idx = np.asarray(comp, dtype=np.int32)
        pts = centers[comp_idx]
        pts_i32 = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        if len(pts_i32) < 3:
            continue

        hull = cv2.convexHull(pts_i32)
        hull_area = float(cv2.contourArea(hull))
        if hull_area < min_hull_area:
            continue

        perimeter = float(cv2.arcLength(hull, True))
        epsilon = max(smooth_eps_base, perimeter * hull_poly_eps_ratio)
        hull_smooth = cv2.approxPolyDP(hull, epsilon, True)
        if len(hull_smooth) < 3:
            hull_smooth = hull

        # 画矩阵区域和边界线
        cv2.drawContours(instance_mask, [hull_smooth], -1, 255, thickness=-1)
        cv2.drawContours(boundary_mask, [hull_smooth], -1, 255, thickness=1, lineType=cv2.LINE_8)
        cv2.drawContours(signature_map, [hull_smooth], -1, 180, thickness=-1)

        # 在当前矩阵内部找边界孔洞点
        boundary_flags = _compute_boundary_holes_in_component(
            comp_points=pts.astype(np.float32),
            hull_for_dist=hull,
            spacing_fallback=spacing,
            line_min_r_ratio=line_min_r_ratio,
            line_max_r_ratio=line_max_r_ratio,
            boundary_band_ratio=boundary_band_ratio,
            boundary_relax_ratio=boundary_relax_ratio,
            boundary_gap_deg=boundary_gap_deg,
        )

        boundary_pts = np.round(pts[boundary_flags]).astype(np.int32)
        boundary_hole_points_list.append(boundary_pts)

        for x, y in boundary_pts:
            if 0 <= x < w and 0 <= y < h:
                boundary_hole_points_mask[y, x] = 255

    return (
        instance_mask,
        boundary_mask,
        signature_map,
        boundary_hole_points_list,
        boundary_hole_points_mask,
    )


def build_boundary_line_mask_from_centers(
    shape: tuple[int, int],
    centers: np.ndarray,
    spacing: float | None = None,
    line_min_r_ratio: float = 0.55,
    line_max_r_ratio: float = 1.65,
    min_component_points: int = 4,
    min_hull_area_ratio: float = 0.20,
    smooth_eps_ratio: float = 0.10,
    smooth_eps_min: float = 1.0,
    hull_poly_eps_ratio: float = 0.0035,
    boundary_band_ratio: float = 0.78,
    boundary_relax_ratio: float = 1.05,
    boundary_gap_deg: float = 115.0,
) -> np.ndarray:
    if len(centers) < 4:
        empty = np.zeros(shape, dtype=np.uint8)
        DEBUG_LAST["instance_mask"] = empty.copy()
        DEBUG_LAST["boundary_mask"] = empty.copy()
        DEBUG_LAST["signature_map"] = empty.copy()
        DEBUG_LAST["boundary_hole_points"] = []
        DEBUG_LAST["boundary_hole_points_mask"] = empty.copy()
        return empty

    tree = cKDTree(centers)
    if spacing is None:
        spacing = estimate_spacing(centers, tree=tree)

    graph = _build_structure_graph(centers, spacing, tree, line_min_r_ratio, line_max_r_ratio)
    components = _connected_components_from_graph(graph)

    instance_mask, boundary_mask, signature_map, boundary_hole_points, boundary_hole_points_mask = (
        _draw_instance_region_mask_convex_hull(
            shape=shape,
            centers=centers,
            components=components,
            spacing=spacing,
            line_min_r_ratio=line_min_r_ratio,
            line_max_r_ratio=line_max_r_ratio,
            min_component_points=min_component_points,
            min_hull_area_ratio=min_hull_area_ratio,
            smooth_eps_ratio=smooth_eps_ratio,
            smooth_eps_min=smooth_eps_min,
            hull_poly_eps_ratio=hull_poly_eps_ratio,
            boundary_band_ratio=boundary_band_ratio,
            boundary_relax_ratio=boundary_relax_ratio,
            boundary_gap_deg=boundary_gap_deg,
        )
    )

    DEBUG_LAST["instance_mask"] = instance_mask
    DEBUG_LAST["boundary_mask"] = boundary_mask
    DEBUG_LAST["signature_map"] = signature_map
    DEBUG_LAST["boundary_hole_points"] = boundary_hole_points
    DEBUG_LAST["boundary_hole_points_mask"] = boundary_hole_points_mask
    return boundary_mask


def _dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)


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
        shape=hole_mask.shape,
        centers=centers,
        line_min_r_ratio=args.line_min_r,
        line_max_r_ratio=args.line_max_r,
        min_component_points=args.min_component_points,
        min_hull_area_ratio=args.min_hull_area_ratio,
        smooth_eps_ratio=args.smooth_eps_ratio,
        smooth_eps_min=args.smooth_eps_min,
        hull_poly_eps_ratio=args.hull_poly_eps_ratio,
        boundary_band_ratio=args.boundary_band_ratio,
        boundary_relax_ratio=args.boundary_relax_ratio,
        boundary_gap_deg=args.boundary_gap_deg,
    )

    boundary_hole_mask = DEBUG_LAST.get("boundary_hole_points_mask")
    if boundary_hole_mask is None:
        boundary_hole_mask = np.zeros_like(line_mask)

    line_vis = _dilate_mask(line_mask, args.preview_dilate_kernel)
    point_vis = _dilate_mask(boundary_hole_mask, args.preview_point_dilate_kernel)

    preview = roi_image.copy()
    preview[line_vis > 0] = (0, 255, 0)
    preview[point_vis > 0] = (0, 0, 255)

    stem = hole_path.stem.replace("_holes_bw", "")
    write_image(output_dir / f"{stem}_pattern_line_preview.png", preview)

    if DEBUG_LAST["instance_mask"] is not None:
        write_image(output_dir / f"{stem}_matrix_instances_bw.png", DEBUG_LAST["instance_mask"])
    if DEBUG_LAST["boundary_mask"] is not None:
        write_image(output_dir / f"{stem}_matrix_boundary_bw.png", DEBUG_LAST["boundary_mask"])
    if DEBUG_LAST["boundary_hole_points_mask"] is not None:
        write_image(output_dir / f"{stem}_boundary_holes_bw.png", DEBUG_LAST["boundary_hole_points_mask"])
    if DEBUG_LAST["signature_map"] is not None:
        write_image(output_dir / f"{stem}_structure_signature_map.png", DEBUG_LAST["signature_map"])


def main() -> None:
    wall_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Find matrix boundaries first, then boundary-hole centers inside each matrix.")
    parser.add_argument("--holes", default="output/holes/3_crop_holes_bw.png", help="Single hole mask file or directory.")
    parser.add_argument("--roi", default="output/roi/3_crop.jpg", help="Corresponding single ROI image or ROI directory.")
    parser.add_argument("--output", default="output/pattern", help="Directory for pattern outputs.")

    parser.add_argument("--preview-dilate-kernel", type=_positive_int, default=3,
                        help="Preview boundary dilation kernel size. Use 1 to disable dilation.")
    parser.add_argument("--preview-point-dilate-kernel", type=_positive_int, default=5,
                        help="Preview boundary-hole dilation kernel size. Use 1 to disable dilation.")

    # --- [1. 孔洞中心提取参数] ---
    group_center = parser.add_argument_group("Center Extraction Parameters")
    group_center.add_argument("--center-min-area", type=_positive_int, default=2)
    group_center.add_argument("--center-max-area", type=_positive_int, default=200)
    group_center.add_argument("--center-max-aspect", type=_positive_float, default=3.0)
    group_center.add_argument("--center-min-fill", type=_ratio_0_1, default=0.2)

    # --- [2. 边界聚类/凸包参数] ---
    group_active = parser.add_argument_group("Boundary Parameters")
    group_active.add_argument("--line-min-r", type=_positive_float, default=0.55,
                              help="Lower neighbor radius ratio based on median spacing.")
    group_active.add_argument("--line-max-r", type=_positive_float, default=1.65,
                              help="Upper neighbor radius ratio based on median spacing.")
    group_active.add_argument("--min-component-points", type=_positive_int, default=4,
                              help="Minimum connected-center count to keep one matrix region.")
    group_active.add_argument("--min-hull-area-ratio", type=_nonnegative_float, default=0.20,
                              help="Minimum hull area ratio relative to spacing^2.")
    group_active.add_argument("--smooth-eps-ratio", type=_nonnegative_float, default=0.10,
                              help="Base contour smoothing ratio relative to spacing.")
    group_active.add_argument("--smooth-eps-min", type=_nonnegative_float, default=1.0,
                              help="Minimum contour smoothing epsilon in pixels.")
    group_active.add_argument("--hull-poly-eps-ratio", type=_nonnegative_float, default=0.0035,
                              help="Douglas-Peucker epsilon ratio relative to hull perimeter.")

    # --- [3. 边界孔洞点参数] ---
    group_bhole = parser.add_argument_group("Boundary Hole Parameters")
    group_bhole.add_argument("--boundary-band-ratio", type=_nonnegative_float, default=0.78,
                             help="Hard distance threshold to hull boundary relative to spacing.")
    group_bhole.add_argument("--boundary-relax-ratio", type=_nonnegative_float, default=1.05,
                             help="Relaxed distance threshold to hull boundary relative to spacing.")
    group_bhole.add_argument("--boundary-gap-deg", type=_nonnegative_float, default=115.0,
                             help="Minimum angular gap for relaxed boundary-hole decision.")

    args = parser.parse_args()
    validate_args(args)

    hole_paths = iter_mask_paths(Path(args.holes))
    if not hole_paths:
        raise RuntimeError(f"No hole masks found in: {args.holes}")

    roi_input = Path(args.roi)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for hole_path in hole_paths:
        roi_path = roi_input if roi_input.is_file() else derive_roi_path(hole_path, roi_input)
        process_path(hole_path, roi_path, output_dir, args)

    print(f"wall: {time.perf_counter() - wall_start:.3f}s")


if __name__ == "__main__":
    main()
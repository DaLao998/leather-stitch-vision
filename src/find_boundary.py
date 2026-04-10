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


DEBUG_LAST: dict[str, np.ndarray | None] = {
    "instance_mask": None,
    "boundary_mask": None,
    "boundary_band_mask": None,
    "signature_map": None,
}


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return ivalue



def _nonnegative_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
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
    if args.band_width_px == 0 and args.band_width_ratio <= 0:
        raise ValueError("band_width_ratio must be > 0 when band_width_px is 0")



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

    # 比逐点 query_ball_point 更省重复查询
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



def _draw_instance_region_mask_convex_hull(
    shape: tuple[int, int],
    centers: np.ndarray,
    components: list[list[int]],
    spacing: float,
    min_component_points: int,
    min_hull_area_ratio: float,
    smooth_eps_ratio: float,
    smooth_eps_min: float,
    hull_poly_eps_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    instance_mask = np.zeros((h, w), dtype=np.uint8)
    boundary_mask = np.zeros((h, w), dtype=np.uint8)
    signature_map = np.zeros((h, w), dtype=np.uint8)

    min_hull_area = max(12.0, (spacing * spacing) * min_hull_area_ratio)
    smooth_eps_base = max(smooth_eps_min, spacing * smooth_eps_ratio)

    for comp in components:
        if len(comp) < min_component_points:
            continue

        pts = centers[np.asarray(comp, dtype=np.int32)]
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

        cv2.drawContours(instance_mask, [hull_smooth], -1, 255, thickness=-1)
        cv2.drawContours(boundary_mask, [hull_smooth], -1, 255, thickness=1, lineType=cv2.LINE_8)

        # 调试图：用每个实例的凸包填充展示
        cv2.drawContours(signature_map, [hull_smooth], -1, 180, thickness=-1)

    return instance_mask, boundary_mask, signature_map



def _resolve_band_width_px(
    spacing: float | None,
    band_width_px: int,
    band_width_ratio: float,
) -> int:
    if band_width_px > 0:
        return band_width_px
    if spacing is None:
        raise ValueError("spacing is required when band_width_px is 0")
    return max(1, int(round(float(spacing) * band_width_ratio)))



def build_boundary_band_mask_from_line_mask(
    line_mask: np.ndarray,
    spacing: float | None = None,
    band_width_px: int = 0,
    band_width_ratio: float = 0.18,
    band_kernel: str = "ellipse",
    band_style: str = "black_on_white",
    frame_width_px: int = 0,
) -> np.ndarray:
    if line_mask.ndim != 2:
        raise ValueError("line_mask must be a 2D binary image")

    if band_kernel == "ellipse":
        kernel_shape = cv2.MORPH_ELLIPSE
    elif band_kernel == "rect":
        kernel_shape = cv2.MORPH_RECT
    elif band_kernel == "cross":
        kernel_shape = cv2.MORPH_CROSS
    else:
        raise ValueError("band_kernel must be 'ellipse', 'rect', or 'cross'")

    if band_width_px > 0:
        width_px = int(band_width_px)
    else:
        if spacing is None:
            raise ValueError("spacing is required when band_width_px is 0")
        width_px = max(1, int(round(float(spacing) * band_width_ratio)))

    if width_px % 2 == 0:
        width_px += 1

    kernel = cv2.getStructuringElement(kernel_shape, (width_px, width_px))

    line_bin = (line_mask > 0).astype(np.uint8) * 255

    # 关键：直接把线膨胀成“实心粗带”
    band_core = cv2.dilate(line_bin, kernel, iterations=1)
    band_core = ((band_core > 0).astype(np.uint8) * 255)

    if band_style == "black_on_white":
        out = np.full(line_mask.shape, 255, dtype=np.uint8)
        out[band_core > 0] = 0
        frame_value = 0
    elif band_style == "white_on_black":
        out = np.zeros(line_mask.shape, dtype=np.uint8)
        out[band_core > 0] = 255
        frame_value = 255
    else:
        raise ValueError("band_style must be 'black_on_white' or 'white_on_black'")

    if frame_width_px > 0:
        fw = int(frame_width_px)
        out[:fw, :] = frame_value
        out[-fw:, :] = frame_value
        out[:, :fw] = frame_value
        out[:, -fw:] = frame_value

    return out


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
) -> np.ndarray:
    if len(centers) < 4:
        empty = np.zeros(shape, dtype=np.uint8)
        DEBUG_LAST["instance_mask"] = empty.copy()
        DEBUG_LAST["boundary_mask"] = empty.copy()
        DEBUG_LAST["boundary_band_mask"] = empty.copy()
        DEBUG_LAST["signature_map"] = empty.copy()
        return empty

    tree = cKDTree(centers)
    if spacing is None:
        spacing = estimate_spacing(centers, tree=tree)

    graph = _build_structure_graph(centers, spacing, tree, line_min_r_ratio, line_max_r_ratio)
    components = _connected_components_from_graph(graph)

    instance_mask, boundary_mask, signature_map = _draw_instance_region_mask_convex_hull(
        shape=shape,
        centers=centers,
        components=components,
        spacing=spacing,
        min_component_points=min_component_points,
        min_hull_area_ratio=min_hull_area_ratio,
        smooth_eps_ratio=smooth_eps_ratio,
        smooth_eps_min=smooth_eps_min,
        hull_poly_eps_ratio=hull_poly_eps_ratio,
    )

    DEBUG_LAST["instance_mask"] = instance_mask
    DEBUG_LAST["boundary_mask"] = boundary_mask
    DEBUG_LAST["boundary_band_mask"] = None
    DEBUG_LAST["signature_map"] = signature_map
    return boundary_mask



def build_boundary_band_mask_from_centers(
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
    band_width_px: int = 0,
    band_width_ratio: float = 0.18,
    band_kernel: str = "ellipse",
    band_style: str = "black_on_white",
    frame_width_px: int = 0,
) -> np.ndarray:
    if len(centers) < 4:
        if band_style == "black_on_white":
            empty = np.full(shape, 255, dtype=np.uint8)
        else:
            empty = np.zeros(shape, dtype=np.uint8)
        DEBUG_LAST["instance_mask"] = np.zeros(shape, dtype=np.uint8)
        DEBUG_LAST["boundary_mask"] = np.zeros(shape, dtype=np.uint8)
        DEBUG_LAST["boundary_band_mask"] = empty.copy()
        DEBUG_LAST["signature_map"] = np.zeros(shape, dtype=np.uint8)
        return empty

    tree = cKDTree(centers)
    if spacing is None:
        spacing = estimate_spacing(centers, tree=tree)

    line_mask = build_boundary_line_mask_from_centers(
        shape=shape,
        centers=centers,
        spacing=spacing,
        line_min_r_ratio=line_min_r_ratio,
        line_max_r_ratio=line_max_r_ratio,
        min_component_points=min_component_points,
        min_hull_area_ratio=min_hull_area_ratio,
        smooth_eps_ratio=smooth_eps_ratio,
        smooth_eps_min=smooth_eps_min,
        hull_poly_eps_ratio=hull_poly_eps_ratio,
    )
    band_mask = build_boundary_band_mask_from_line_mask(
        line_mask=line_mask,
        spacing=spacing,
        band_width_px=band_width_px,
        band_width_ratio=band_width_ratio,
        band_kernel=band_kernel,
        band_style=band_style,
        frame_width_px=frame_width_px,
    )
    DEBUG_LAST["boundary_band_mask"] = band_mask
    return band_mask



def build_boundary_mask_from_centers(
    shape: tuple[int, int],
    centers: np.ndarray,
    output_kind: str = "line",
    spacing: float | None = None,
    line_min_r_ratio: float = 0.55,
    line_max_r_ratio: float = 1.65,
    min_component_points: int = 4,
    min_hull_area_ratio: float = 0.20,
    smooth_eps_ratio: float = 0.10,
    smooth_eps_min: float = 1.0,
    hull_poly_eps_ratio: float = 0.0035,
    band_width_px: int = 0,
    band_width_ratio: float = 0.18,
    band_kernel: str = "ellipse",
    band_style: str = "black_on_white",
    frame_width_px: int = 0,
) -> np.ndarray:
    if spacing is None and len(centers) >= 2:
        spacing = estimate_spacing(centers)

    line_mask = build_boundary_line_mask_from_centers(
        shape=shape,
        centers=centers,
        spacing=spacing,
        line_min_r_ratio=line_min_r_ratio,
        line_max_r_ratio=line_max_r_ratio,
        min_component_points=min_component_points,
        min_hull_area_ratio=min_hull_area_ratio,
        smooth_eps_ratio=smooth_eps_ratio,
        smooth_eps_min=smooth_eps_min,
        hull_poly_eps_ratio=hull_poly_eps_ratio,
    )

    if output_kind == "line":
        return line_mask
    if output_kind == "band":
        instance_mask = DEBUG_LAST["instance_mask"]
        if instance_mask is None:
            if band_style == "black_on_white":
                return np.full(shape, 255, dtype=np.uint8)
            return np.zeros(shape, dtype=np.uint8)
        return instance_mask.copy()

    raise ValueError("output_kind must be 'line' or 'band'")


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

    spacing = estimate_spacing(centers)
    line_mask = build_boundary_line_mask_from_centers(
        shape=hole_mask.shape,
        centers=centers,
        spacing=spacing,
        line_min_r_ratio=args.line_min_r,
        line_max_r_ratio=args.line_max_r,
        min_component_points=args.min_component_points,
        min_hull_area_ratio=args.min_hull_area_ratio,
        smooth_eps_ratio=args.smooth_eps_ratio,
        smooth_eps_min=args.smooth_eps_min,
        hull_poly_eps_ratio=args.hull_poly_eps_ratio,
    )
    band_mask = build_boundary_band_mask_from_line_mask(
        line_mask=line_mask,
        spacing=spacing,
        band_width_px=args.band_width_px,
        band_width_ratio=args.band_width_ratio,
        band_kernel=args.band_kernel,
        band_style=args.band_style,
        frame_width_px=args.band_frame_px,
    )
    DEBUG_LAST["boundary_band_mask"] = band_mask

    preview = roi_image.copy()
    preview[line_mask > 0] = (0, 255, 0)

    band_preview = roi_image.copy()
    if args.band_style == "black_on_white":
        band_pixels = band_mask == 0
    else:
        band_pixels = band_mask > 0
    band_preview[band_pixels] = (0, 255, 255)

    stem = hole_path.stem.replace("_holes_bw", "")
    write_image(output_dir / f"{stem}_pattern_line_preview.png", preview)
    write_image(output_dir / f"{stem}_pattern_band_preview.png", band_preview)

    if DEBUG_LAST["instance_mask"] is not None:
        write_image(output_dir / f"{stem}_matrix_instances_bw.png", DEBUG_LAST["instance_mask"])
    if DEBUG_LAST["boundary_mask"] is not None:
        write_image(output_dir / f"{stem}_matrix_boundary_bw.png", DEBUG_LAST["boundary_mask"])
    if DEBUG_LAST["boundary_band_mask"] is not None:
        write_image(output_dir / f"{stem}_matrix_boundary_band_bw.png", DEBUG_LAST["boundary_band_mask"])
    if DEBUG_LAST["signature_map"] is not None:
        write_image(output_dir / f"{stem}_structure_signature_map.png", DEBUG_LAST["signature_map"])



def main() -> None:
    wall_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Find matrix-instance boundaries using convex hull on XY clusters.")
    parser.add_argument("--holes", default="output/holes/3_crop_holes_bw.png", help="Single hole mask file or directory.")
    parser.add_argument("--roi", default="output/roi/3_crop.jpg", help="Corresponding single ROI image or ROI directory.")
    parser.add_argument("--output", default="output/pattern", help="Directory for pattern outputs.")

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

    # --- [3. 带状二值图参数] ---
    group_band = parser.add_argument_group("Band Mask Parameters")
    group_band.add_argument("--band-width-px", type=_nonnegative_int, default=0,
                            help="Direct band width in pixels. If 0, use band-width-ratio * spacing.")
    group_band.add_argument("--band-width-ratio", type=_nonnegative_float, default=0.18,
                            help="Band width ratio relative to median spacing when band-width-px is 0.")
    group_band.add_argument("--band-kernel", choices=["ellipse", "rect", "cross"], default="ellipse",
                            help="Structuring element shape used to thicken the boundary.")
    group_band.add_argument("--band-style", choices=["black_on_white", "white_on_black"], default="black_on_white",
                            help="Output polarity of the band binary image.")
    group_band.add_argument("--band-frame-px", type=_nonnegative_int, default=0,
                            help="Optional outer frame width in pixels.")

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

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from src.crop_seat_roi import build_roi_crop
from src.extract_holes import build_hole_mask_and_centers
from utils.find_boundary_hole import build_boundary_line_mask_from_centers, DEBUG_LAST
from src.utils import iter_image_paths, load_bgr_image, write_image

# 黄色坐垫（图片大）
# ==============================================================================
# 全局超参数配置区 (Global Hyperparameters)
# 您可以在这里直接修改默认参数，或通过命令行覆盖
# ==============================================================================

# --- [0. 全局预处理参数] ---
IMAGE_SCALE = 1.0
PREVIEW_DILATE_KERNEL = 3
PREVIEW_POINT_DILATE_KERNEL = 5

# --- [1. crop_seat_roi.py: 区域裁剪参数] ---
BASE_IMAGE_WIDTH = 5472
BASE_IMAGE_HEIGHT = 3648
ROI_POINTS = [1689, 201, 3214, 228, 3180, 2445, 1584, 2382]

# --- [2. extract_holes.py: 打孔提取参数] ---
ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 7
CLEAN_KERNEL_SIZE = 6
MERGE_KERNEL_SIZE = 5
HOLE_MIN_AREA = 6
HOLE_MAX_AREA = 120
HOLE_MAX_ASPECT_RATIO = 1.8
HOLE_MIN_FILL_RATIO = 0.5

# --- [3. find_boundary.py: 边界聚类/凸包参数] ---
LINE_MIN_R_RATIO = 0.70
LINE_MAX_R_RATIO = 1.55
MIN_COMPONENT_POINTS = 4
MIN_HULL_AREA_RATIO = 0.20
SMOOTH_EPS_RATIO = 0.10
SMOOTH_EPS_MIN = 1.0
HULL_POLY_EPS_RATIO = 0.0035

# --- [4. 边界孔洞点参数] ---
BOUNDARY_BAND_RATIO = 0.78        # 到 hull 边界的硬阈值（spacing 倍数）
BOUNDARY_RELAX_RATIO = 1.05       # 放宽距离阈值（需配合角度缺口）
BOUNDARY_GAP_DEG = 115.0          # 邻居最大角度缺口阈值（度）


# ==============================================================================


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return ivalue


def _positive_float(value: str) -> float:
    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return fvalue


def _nonnegative_float(value: str) -> float:
    fvalue = float(value)
    if fvalue < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return fvalue


def _ratio_0_1(value: str) -> float:
    fvalue = float(value)
    if not (0.0 <= fvalue <= 1.0):
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return fvalue


def _odd_positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0 or ivalue % 2 == 0:
        raise argparse.ArgumentTypeError("must be a positive odd integer")
    return ivalue


def validate_args(args: argparse.Namespace) -> None:
    if args.base_width <= 0 or args.base_height <= 0:
        raise ValueError("base_width and base_height must be > 0")
    if len(args.roi_points) != 8:
        raise ValueError("roi_points must contain exactly 8 numbers")
    if args.hole_min_area > args.hole_max_area:
        raise ValueError("hole_min_area cannot be larger than hole_max_area")
    if args.line_min_r >= args.line_max_r:
        raise ValueError("line_min_r must be smaller than line_max_r")
    if args.boundary_band_ratio > args.boundary_relax_ratio:
        raise ValueError("boundary_band_ratio cannot be larger than boundary_relax_ratio")


def _dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)


def process_path(image_path: Path, output_dir: Path, args: argparse.Namespace) -> tuple[float, float, float]:
    image = load_bgr_image(image_path)

    # 0. 初始缩放
    if args.image_scale != 1.0:
        image = cv2.resize(image, None, fx=args.image_scale, fy=args.image_scale, interpolation=cv2.INTER_NEAREST)

    # 1. 裁剪 ROI
    t1_start = time.perf_counter()
    crop = build_roi_crop(
        image,
        points=args.roi_points,
        base_width=args.base_width,
        base_height=args.base_height,
    )
    elapsed1 = time.perf_counter() - t1_start

    # 2. 提取打孔与中心点
    t2_start = time.perf_counter()
    _, hole_mask, centers = build_hole_mask_and_centers(
        crop,
        min_area=args.hole_min_area,
        max_area=args.hole_max_area,
        max_aspect_ratio=args.hole_max_aspect,
        min_fill_ratio=args.hole_min_fill,
        adaptive_block_size=args.adaptive_block_size,
        adaptive_c=args.adaptive_c,
        clean_kernel_size=args.clean_kernel,
        merge_kernel_size=args.merge_kernel,
    )
    elapsed2 = time.perf_counter() - t2_start

    # 3. 用 hull 找矩阵，再找边界孔洞点
    t3_start = time.perf_counter()
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

    pattern_preview = crop.copy()
    pattern_preview[line_vis > 0] = (0, 255, 0)      # 绿线：矩阵边界
    pattern_preview[point_vis > 0] = (0, 0, 255)     # 红点：边界孔洞点

    elapsed3 = time.perf_counter() - t3_start

    # 保存结果
    stem = image_path.stem
    write_image(output_dir / f"{stem}_pattern_preview.png", pattern_preview)

    if DEBUG_LAST.get("instance_mask") is not None:
        write_image(output_dir / f"{stem}_matrix_instances_bw.png", DEBUG_LAST["instance_mask"])
    if DEBUG_LAST.get("boundary_mask") is not None:
        write_image(output_dir / f"{stem}_matrix_boundary_bw.png", DEBUG_LAST["boundary_mask"])
    if DEBUG_LAST.get("boundary_hole_points_mask") is not None:
        write_image(output_dir / f"{stem}_boundary_holes_bw.png", DEBUG_LAST["boundary_hole_points_mask"])
    if DEBUG_LAST.get("signature_map") is not None:
        write_image(output_dir / f"{stem}_structure_signature_map.png", DEBUG_LAST["signature_map"])

    return elapsed1, elapsed2, elapsed3


def main() -> None:
    total_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run ROI crop, hole extraction, and boundary-hole preview generation.")

    # 基础路径参数
    parser.add_argument("--input", default="picture/3.jpg", help="Image file or directory.")
    parser.add_argument("--output", default="output/pattern1", help="Directory for final preview images.")
    parser.add_argument("--image-scale", type=_positive_float, default=IMAGE_SCALE, help="Global image scale.")
    parser.add_argument(
        "--preview-dilate-kernel",
        type=_positive_int,
        default=PREVIEW_DILATE_KERNEL,
        help="Preview boundary dilation kernel size. Use 1 to disable dilation.",
    )
    parser.add_argument(
        "--preview-point-dilate-kernel",
        type=_positive_int,
        default=PREVIEW_POINT_DILATE_KERNEL,
        help="Preview boundary-hole dilation kernel size. Use 1 to disable dilation.",
    )

    # 1. crop ROI 组
    group_roi = parser.add_argument_group("ROI Crop Parameters")
    group_roi.add_argument("--base-width", type=_positive_int, default=BASE_IMAGE_WIDTH)
    group_roi.add_argument("--base-height", type=_positive_int, default=BASE_IMAGE_HEIGHT)
    group_roi.add_argument("--roi-points", type=float, nargs=8, default=ROI_POINTS)

    # 2. extract holes 组
    group_hole = parser.add_argument_group("Hole Extraction Parameters")
    group_hole.add_argument("--adaptive-block-size", type=_odd_positive_int, default=ADAPTIVE_BLOCK_SIZE)
    group_hole.add_argument("--adaptive-c", type=float, default=ADAPTIVE_C)
    group_hole.add_argument("--clean-kernel", type=_positive_int, default=CLEAN_KERNEL_SIZE)
    group_hole.add_argument("--merge-kernel", type=_positive_int, default=MERGE_KERNEL_SIZE)
    group_hole.add_argument("--hole-min-area", type=_positive_int, default=HOLE_MIN_AREA)
    group_hole.add_argument("--hole-max-area", type=_positive_int, default=HOLE_MAX_AREA)
    group_hole.add_argument("--hole-max-aspect", type=_positive_float, default=HOLE_MAX_ASPECT_RATIO)
    group_hole.add_argument("--hole-min-fill", type=_ratio_0_1, default=HOLE_MIN_FILL_RATIO)

    # 3. boundary 组
    group_bound = parser.add_argument_group("Active Boundary Parameters")
    group_bound.add_argument("--line-min-r", type=_positive_float, default=LINE_MIN_R_RATIO)
    group_bound.add_argument("--line-max-r", type=_positive_float, default=LINE_MAX_R_RATIO)
    group_bound.add_argument("--min-component-points", type=_positive_int, default=MIN_COMPONENT_POINTS)
    group_bound.add_argument("--min-hull-area-ratio", type=_nonnegative_float, default=MIN_HULL_AREA_RATIO)
    group_bound.add_argument("--smooth-eps-ratio", type=_nonnegative_float, default=SMOOTH_EPS_RATIO)
    group_bound.add_argument("--smooth-eps-min", type=_nonnegative_float, default=SMOOTH_EPS_MIN)
    group_bound.add_argument("--hull-poly-eps-ratio", type=_nonnegative_float, default=HULL_POLY_EPS_RATIO)

    # 4. boundary-hole 组
    group_bhole = parser.add_argument_group("Boundary Hole Parameters")
    group_bhole.add_argument("--boundary-band-ratio", type=_nonnegative_float, default=BOUNDARY_BAND_RATIO)
    group_bhole.add_argument("--boundary-relax-ratio", type=_nonnegative_float, default=BOUNDARY_RELAX_RATIO)
    group_bhole.add_argument("--boundary-gap-deg", type=_nonnegative_float, default=BOUNDARY_GAP_DEG)

    args = parser.parse_args()
    validate_args(args)

    image_paths = iter_image_paths(Path(args.input))
    if not image_paths:
        raise RuntimeError(f"No images found under: {args.input}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total1, total2, total3 = 0.0, 0.0, 0.0

    for image_path in image_paths:
        e1, e2, e3 = process_path(image_path, output_dir, args)
        total1 += e1
        total2 += e2
        total3 += e3

    total = total1 + total2 + total3
    total_time = time.perf_counter() - total_start
    print(f"耗时1: {total1:.3f}s + 耗时2: {total2:.3f}s + 耗时3: {total3:.3f}s = 纯算法总耗时: {total:.3f}s")
    print(f"程序墙钟总耗时：{total_time:.3f} s")


if __name__ == "__main__":
    main()
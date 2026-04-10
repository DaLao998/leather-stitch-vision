from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from src.crop_seat_roi import build_roi_crop
from src.extract_holes import build_hole_mask_and_centers
from src.find_boundary import build_boundary_mask_from_centers
from src.utils import iter_image_paths, load_bgr_image, write_image

# 黄色坐垫（图片大）
# ==============================================================================
# 全局超参数配置区 (Global Hyperparameters)             
# 您可以在这里直接修改默认参数，或通过命令行覆盖
# ==============================================================================

# --- [0. 全局预处理参数] ---
IMAGE_SCALE = 1.0               # 图像初始缩放比例 (例如 0.5 为缩小一半，可加速处理)
PREVIEW_DILATE_KERNEL = 1         # 预览图中边界绿线的膨胀核大小，建议奇数；<=1 表示不膨胀

# --- [1. crop_seat_roi.py: 区域裁剪参数] ---
BASE_IMAGE_WIDTH = 5472             # 基准图像宽度 (用于将ROI坐标等比映射到实际图片)
BASE_IMAGE_HEIGHT = 3648            # 基准图像高度
ROI_POINTS = [1689, 201, 3214, 228, 3180, 2445, 1584, 2382]  # ROI多边形顶点坐标 [x1,y1, x2,y2, x3,y3, x4,y4]

# --- [2. extract_holes.py: 打孔提取参数] ---
# 自适应阈值提取
ADAPTIVE_BLOCK_SIZE = 31     # 局部观察窗口大小(必须是奇数)。截取孔洞越大该值需要越大
ADAPTIVE_C = 7               # 灵敏度。数字越小，提取出的周围散乱噪点（缝纫线等）越丰富
# 形态学操作
CLEAN_KERNEL_SIZE = 6        # 过滤孤立小噪点的椭圆核大小
MERGE_KERNEL_SIZE = 5        # 强制粘连杂乱密集噪点的椭圆核大小
# 连通域过滤 (剥离非孔洞杂质)
HOLE_MIN_AREA = 6            # 最小合法孔洞面积
HOLE_MAX_AREA = 120          # 最大合法孔洞面积
HOLE_MAX_ASPECT_RATIO = 1.8  # 最大长宽比限制 (过滤长条形噪点)
HOLE_MIN_FILL_RATIO = 0.5    # 最小填充率限制 (面积/矩形框面积，过滤不规则图形)

# --- [3. find_boundary_hole.py: 边界聚类/凸包参数] ---
LINE_MIN_R_RATIO = 0.70           # 连边搜索最小半径系数（基于点距中位数）
LINE_MAX_R_RATIO = 1.55           # 连边搜索最大半径系数（基于点距中位数）
MIN_COMPONENT_POINTS = 4          # 一个孔洞阵列最少需要多少个中心点才保留
MIN_HULL_AREA_RATIO = 0.20        # 阵列凸包最小面积阈值（spacing^2 的倍数）
SMOOTH_EPS_RATIO = 0.10           # 平滑基线系数（spacing 的倍数）
SMOOTH_EPS_MIN = 1.0              # 平滑最小 epsilon（像素）
HULL_POLY_EPS_RATIO = 0.0035      # 轮廓简化 epsilon（perimeter 的倍数）
# 输出图模式
LINE = "band"
BAND_WIDTH_PX = 0
BAND_WIDTH_RATIO = 0.18
BAND_KERNEL = "ellipse"
BAND_STYLE = "black_on_white"
BAND_FRAME_PX = 0

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

    # 3. 计算边界连线并生成预览图
    t3_start = time.perf_counter()
    output_mask = build_boundary_mask_from_centers(
        shape=hole_mask.shape,
        centers=centers,
        output_kind=args.output_kind,
        line_min_r_ratio=args.line_min_r,
        line_max_r_ratio=args.line_max_r,
        min_component_points=args.min_component_points,
        min_hull_area_ratio=args.min_hull_area_ratio,
        smooth_eps_ratio=args.smooth_eps_ratio,
        smooth_eps_min=args.smooth_eps_min,
        hull_poly_eps_ratio=args.hull_poly_eps_ratio,
        band_width_px=args.band_width_px,
        band_width_ratio=args.band_width_ratio,
        band_kernel=args.band_kernel,
        band_style=args.band_style,
        frame_width_px=args.band_frame_px,
    )

    if args.output_kind == "line":
        # 膨胀绿线使其在预览图中更显眼
        if args.preview_dilate_kernel > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (args.preview_dilate_kernel, args.preview_dilate_kernel)
            )
            output_mask = cv2.dilate(output_mask, kernel, iterations=1)

        pattern_preview = crop.copy()
        pattern_preview[output_mask > 0] = (0, 255, 0)
    else:
        pattern_preview = output_mask
    elapsed3 = time.perf_counter() - t3_start

    # 保存结果
    stem = image_path.stem
    if args.output_kind == "line":
        write_image(output_dir / f"{stem}_pattern_preview.png", pattern_preview)
    else:
        write_image(output_dir / f"{stem}_matrix_instances_bw.png", pattern_preview)

    return elapsed1, elapsed2, elapsed3



def main() -> None:
    total_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run ROI crop, hole extraction, and boundary preview generation.")

    # 基础路径参数
    parser.add_argument("--input", default="picture/1.jpg", help="Image file or directory.")
    parser.add_argument("--output", default="output/pattern", help="Directory for final preview images.")
    parser.add_argument("--image-scale", type=_positive_float, default=IMAGE_SCALE, help="Global image scale.")
    parser.add_argument(
        "--preview-dilate-kernel",
        type=_positive_int,
        default=PREVIEW_DILATE_KERNEL,
        help="Preview boundary dilation kernel size. Use 1 to disable dilation.",
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
    group_bound.add_argument("--output-kind", choices=["line", "band"], default=LINE)
    group_bound.add_argument("--band-width-px", type=int, default=BAND_WIDTH_PX)
    group_bound.add_argument("--band-width-ratio", type=float, default=BAND_WIDTH_RATIO)
    group_bound.add_argument("--band-kernel", choices=["ellipse", "rect", "cross"], default=BAND_KERNEL)
    group_bound.add_argument("--band-style", choices=["black_on_white", "white_on_black"], default=BAND_STYLE)
    group_bound.add_argument("--band-frame-px", type=int, default=BAND_FRAME_PX)


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

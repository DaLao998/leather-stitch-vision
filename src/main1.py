from __future__ import annotations

import argparse
import time
from pathlib import Path
import cv2

from src.crop_seat_roi import build_roi_crop
from src.extract_holes import build_hole_mask_and_centers
from src.find_boundary_hole import build_boundary_line_mask_from_centers
from src.utils import iter_image_paths, load_bgr_image, write_image

# 灰色坐垫（图片小）
# ==============================================================================
# 全局超参数配置区 (Global Hyperparameters)             
# 您可以在这里直接修改默认参数，或通过命令行覆盖
# ==============================================================================

# --- [0. 全局预处理参数] ---
IMAGE_SCALE = 1.0               # 图像初始缩放比例 (例如 0.5 为缩小一半，可加速处理)

# --- [1. crop_seat_roi.py: 区域裁剪参数] ---  
BASE_IMAGE_WIDTH = 898         # 基准图像宽度 (用于将ROI坐标等比映射到实际图片)
BASE_IMAGE_HEIGHT = 969        # 基准图像高度
ROI_POINTS = [0, 0, 0, 969, 898, 969, 898, 0] # ROI多边形顶点坐标 [x1,y1, x2,y2, x3,y3, x4,y4]

# --- [2. extract_holes.py: 打孔提取参数] ---
# 自适应阈值提取
ADAPTIVE_BLOCK_SIZE = 31        # 局部观察窗口大小(必须是奇数)。截取孔洞越大该值需要越大
ADAPTIVE_C = 7                  # 灵敏度。数字越小，提取出的周围散乱噪点（缝纫线等）越丰富
# 形态学操作
CLEAN_KERNEL_SIZE = 3           # 过滤孤立小噪点的椭圆核大小
MERGE_KERNEL_SIZE = 5           # 强制粘连杂乱密集噪点的椭圆核大小
# 连通域过滤 (剥离非孔洞杂质)
HOLE_MIN_AREA = 6               # 最小合法孔洞面积
HOLE_MAX_AREA = 120             # 最大合法孔洞面积
HOLE_MAX_ASPECT_RATIO = 1.8     # 最大长宽比限制 (过滤长条形噪点)
HOLE_MIN_FILL_RATIO = 0.5       # 最小填充率限制 (面积/矩形框面积，过滤不规则图形)

# --- [3. find_boundary_hole.py: 边界孔洞寻找参数] ---
# 孔洞中心二次过滤 (计算边界前剔除异常点)
CENTER_MIN_AREA = 2             # 提取中心点时的最小连通域面积
CENTER_MAX_AREA = 200           # 提取中心点时的最大连通域面积
CENTER_MAX_ASPECT = 3.0         # 提取中心点时的最大长宽比
CENTER_MIN_FILL = 0.2           # 提取中心点时的最小填充率
# 边界点判定逻辑
BOUNDARY_NEIGHBOR_MAX = 4       # 判定为边界的最大邻居数量阈值 (<=该值视为边界)
BOUNDARY_SECTORS_MAX = 5        # 判定为边界的最多被占用扇区数量 (总8扇区，<=该值视为边界)
BOUNDARY_GAP_DEG_MIN = 115.0    # 判定为边界的最小最大张角 (>=该角度视为边界)
# 边界点连线逻辑
LINE_MIN_R_RATIO = 0.70         # 连线时搜索邻居的最小半径系数 (基于点距中位数的倍数)
LINE_MAX_R_RATIO = 1.55         # 连线时搜索邻居的最大半径系数 (基于点距中位数的倍数)

# ==============================================================================

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
        base_height=args.base_height
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
    line_mask = build_boundary_line_mask_from_centers(
        shape=hole_mask.shape,
        centers=centers,
        center_min_area=args.center_min_area,
        center_max_area=args.center_max_area,
        center_max_aspect=args.center_max_aspect,
        center_min_fill=args.center_min_fill,
        neighbor_count_max=args.boundary_neighbor_max,
        occupied_sectors_max=args.boundary_sectors_max,
        max_gap_deg_min=args.boundary_gap_min,
        line_min_r_ratio=args.line_min_r,
        line_max_r_ratio=args.line_max_r,
    )
    
    # 膨胀绿线使其在预览图中更显眼
    line_mask = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    pattern_preview = crop.copy()
    pattern_preview[line_mask > 0] = (0, 255, 0)
    elapsed3 = time.perf_counter() - t3_start

    # 保存结果
    stem = image_path.stem
    write_image(output_dir / f"{stem}_pattern_preview.png", pattern_preview)

    return elapsed1, elapsed2, elapsed3

def main() -> None:
    total_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run ROI crop, hole extraction, and boundary preview generation.")
    
    # 基础路径参数
    parser.add_argument("--input", default="picture/4.png", help="Image file or directory.")
    parser.add_argument("--output", default="output/pattern", help="Directory for final preview images.")
    parser.add_argument("--image-scale", type=float, default=IMAGE_SCALE, help="Global image scale.")

    # 1. crop ROI 组
    group_roi = parser.add_argument_group("ROI Crop Parameters")
    group_roi.add_argument("--base-width", type=int, default=BASE_IMAGE_WIDTH)
    group_roi.add_argument("--base-height", type=int, default=BASE_IMAGE_HEIGHT)
    group_roi.add_argument("--roi-points", type=float, nargs=8, default=ROI_POINTS)

    # 2. extract holes 组
    group_hole = parser.add_argument_group("Hole Extraction Parameters")
    group_hole.add_argument("--adaptive-block-size", type=int, default=ADAPTIVE_BLOCK_SIZE)
    group_hole.add_argument("--adaptive-c", type=int, default=ADAPTIVE_C)
    group_hole.add_argument("--clean-kernel", type=int, default=CLEAN_KERNEL_SIZE)
    group_hole.add_argument("--merge-kernel", type=int, default=MERGE_KERNEL_SIZE)
    group_hole.add_argument("--hole-min-area", type=int, default=HOLE_MIN_AREA)
    group_hole.add_argument("--hole-max-area", type=int, default=HOLE_MAX_AREA)
    group_hole.add_argument("--hole-max-aspect", type=float, default=HOLE_MAX_ASPECT_RATIO)
    group_hole.add_argument("--hole-min-fill", type=float, default=HOLE_MIN_FILL_RATIO)

    # 3. boundary holes 组
    group_bound = parser.add_argument_group("Boundary Detection Parameters")
    group_bound.add_argument("--center-min-area", type=int, default=CENTER_MIN_AREA)
    group_bound.add_argument("--center-max-area", type=int, default=CENTER_MAX_AREA)
    group_bound.add_argument("--center-max-aspect", type=float, default=CENTER_MAX_ASPECT)
    group_bound.add_argument("--center-min-fill", type=float, default=CENTER_MIN_FILL)
    group_bound.add_argument("--boundary-neighbor-max", type=int, default=BOUNDARY_NEIGHBOR_MAX)
    group_bound.add_argument("--boundary-sectors-max", type=int, default=BOUNDARY_SECTORS_MAX)
    group_bound.add_argument("--boundary-gap-min", type=float, default=BOUNDARY_GAP_DEG_MIN)
    group_bound.add_argument("--line-min-r", type=float, default=LINE_MIN_R_RATIO)
    group_bound.add_argument("--line-max-r", type=float, default=LINE_MAX_R_RATIO)

    args = parser.parse_args()

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
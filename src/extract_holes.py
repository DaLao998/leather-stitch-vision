from __future__ import annotations

import argparse
import time
from pathlib import Path
import cv2
import numpy as np

from src.utils import iter_image_paths, load_bgr_image, write_image

def build_hole_mask_and_centers(
    image: np.ndarray,
    min_area: int = 6,
    max_area: int = 120,
    max_aspect_ratio: float = 1.8,
    min_fill_ratio: float = 0.5,
    adaptive_block_size: int = 31,
    adaptive_c: int = 7,
    clean_kernel_size: int = 6,
    merge_kernel_size: int = 5,
    discard_border_components: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. 自适应阈值直接提取
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=adaptive_block_size,
        C=adaptive_c
    )
    
    response = binary.copy()  # 用于 debug 输出带有所有原始噪点的图

    # === 移除原本在这里的开运算，直接进行闭操作建桥 ===

    # 2. 将杂乱密集的噪点强制粘连，形成不规则怪物
    merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_kernel_size, merge_kernel_size))
    merged_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, merge_kernel)

    # 3. 计算连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(merged_binary, connectivity=8)

    areas = stats[:, cv2.CC_STAT_AREA].astype(np.int32)
    widths = stats[:, cv2.CC_STAT_WIDTH].astype(np.int32)
    heights = stats[:, cv2.CC_STAT_HEIGHT].astype(np.int32)

    short_side = np.maximum(1, np.minimum(widths, heights)).astype(np.float32)
    long_side = np.maximum(widths, heights).astype(np.float32)
    aspect_ratio = long_side / short_side
    fill_ratio = areas.astype(np.float32) / np.maximum(1, widths * heights).astype(np.float32)

    # 4. 过滤掉被粘连的怪物噪点
    keep = (
        (areas >= int(min_area))
        & (areas <= int(max_area))
        & (aspect_ratio <= float(max_aspect_ratio))
        & (fill_ratio >= float(min_fill_ratio))
    )
    keep[0] = False

    if discard_border_components and num_labels > 1:
        border_ids = np.unique(np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1])))
        border_mask = np.zeros(num_labels, dtype=bool)
        border_mask[border_ids] = True
        keep &= ~border_mask
        keep[0] = False

    valid_regions_mask = (keep[labels].astype(np.uint8) * 255)

    # 5. 还原孔洞的原始像素
    hole_mask = cv2.bitwise_and(binary, valid_regions_mask)

    # === 核心修改点：把开运算挪到这里！作为终极抛光 ===
    # 过滤可能残留的孤立小噪点以及孔洞边缘的毛刺
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_kernel_size, clean_kernel_size))
    hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_OPEN, clean_kernel)
    # ==================================================

    # 计算最终中心点
    num_final, _, _, final_centroids = cv2.connectedComponentsWithStats(hole_mask, connectivity=8)
    centers = final_centroids[1:].astype(np.float32) if num_final > 1 else np.empty((0, 2), dtype=np.float32)

    return response, hole_mask, centers

def build_preview_image(image: np.ndarray, hole_mask: np.ndarray) -> np.ndarray:
    preview = image.copy()
    preview[hole_mask > 0] = 255
    return preview

def process_path(image_path: Path, output_dir: Path, args: argparse.Namespace) -> None:
    image = load_bgr_image(image_path)

    response, hole_mask, _ = build_hole_mask_and_centers(
        image,
        min_area=args.min_area,
        max_area=args.max_area,
        max_aspect_ratio=args.max_aspect_ratio,
        min_fill_ratio=args.min_fill_ratio,
        adaptive_block_size=args.adaptive_block_size,
        adaptive_c=args.adaptive_c,
        clean_kernel_size=args.clean_kernel_size,
        merge_kernel_size=args.merge_kernel_size,
        discard_border_components=args.discard_border_components,
    )

    stem = image_path.stem
    write_image(output_dir / f"{stem}_holes_bw.png", hole_mask)

    if args.save_debug or args.save_response:
        write_image(output_dir / f"{stem}_threshold_raw.png", response)
    if args.save_debug or args.save_inverted:
        write_image(output_dir / f"{stem}_holes_bw_inverted.png", cv2.bitwise_not(hole_mask))
    if args.save_debug or args.save_preview:
        write_image(output_dir / f"{stem}_holes_preview.png", build_preview_image(image, hole_mask))

def main() -> None:
    wall_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Extract perforation holes into a binary image.")
    # 单独运行时，默认输入设为单图
    parser.add_argument("--input", default="output/roi/3_crop.jpg", help="Target Image file.")
    parser.add_argument("--output", default="output/holes", help="Directory for result images.")
    
    # 超参数与主程序保持一致
    parser.add_argument("--min-area", type=int, default=6)
    parser.add_argument("--max-area", type=int, default=120)
    parser.add_argument("--max-aspect-ratio", type=float, default=1.8)
    parser.add_argument("--min-fill-ratio", type=float, default=0.5)
    parser.add_argument("--adaptive-block-size", type=int, default=31)
    parser.add_argument("--adaptive-c", type=int, default=7)
    parser.add_argument("--clean-kernel-size", type=int, default=6)
    parser.add_argument("--merge-kernel-size", type=int, default=5)
    
    parser.add_argument("--discard-border-components", action="store_true")
    parser.add_argument("--save-debug", action="store_true", help="Save all intermediate debug images.")
    parser.add_argument("--save-response", action="store_true")
    parser.add_argument("--save-inverted", action="store_true")
    parser.add_argument("--save-preview", action="store_true")
    args = parser.parse_args()

    image_paths = iter_image_paths(Path(args.input))
    if not image_paths:
        raise RuntimeError(f"No images found under: {args.input}")

    output_dir = Path(args.output)
    for image_path in image_paths:
        process_path(image_path, output_dir, args)

    print(f"wall: {time.perf_counter() - wall_start:.3f}s")

if __name__ == "__main__":
    main()
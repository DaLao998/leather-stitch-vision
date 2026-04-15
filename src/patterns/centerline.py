import time
from pathlib import Path

import cv2
import numpy as np


def get_crop_local_polygon():
    """ROI polygon in original image coordinates. Used only for optional debug drawing."""
    return np.array(
        [
            [105, 0],
            [1630, 27],
            [1596, 2244],
            [0, 2181],
        ],
        dtype=np.int32,
    )


def _scaled_odd_kernel(value, scale, minimum=3):
    size = max(minimum, int(round(value * scale)))
    return size + 1 if size % 2 == 0 else size


def _resize_mask_to_original(mask, original_shape):
    height, width = original_shape[:2]
    if mask.shape == (height, width):
        return mask
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def _centerline_between_white_regions(
    binary,
    min_radius,
    max_radius,
    probe_scale=1.35,
    probe_angles=8,
    reconnect_radius=35,
):
    """
    Build a curved centerline mask from a black-on-white binary image.

    The skeleton itself also contains centerlines for the outer black frame.
    To reject those, keep skeleton pixels that can see white pixels on two
    opposite sides. A small reconstruction step fills junctions that fail that
    local two-side test but are adjacent to valid centerline pixels.
    """
    skeleton = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    candidate = (skeleton > 0) & (dist >= min_radius) & (dist <= max_radius)
    ys, xs = np.where(candidate)
    if len(xs) == 0:
        return np.zeros_like(binary), skeleton

    height, width = binary.shape[:2]
    radii = np.clip(dist[ys, xs] * probe_scale, max(5.0, min_radius), max(6.0, max_radius))
    keep = np.zeros(len(xs), dtype=bool)

    for angle in np.linspace(0, np.pi, probe_angles, endpoint=False):
        dx = np.rint(np.cos(angle) * radii).astype(np.int32)
        dy = np.rint(np.sin(angle) * radii).astype(np.int32)

        x1 = np.clip(xs + dx, 0, width - 1)
        y1 = np.clip(ys + dy, 0, height - 1)
        x2 = np.clip(xs - dx, 0, width - 1)
        y2 = np.clip(ys - dy, 0, height - 1)

        keep |= (binary[y1, x1] == 0) & (binary[y2, x2] == 0)

    seed = np.zeros_like(binary)
    seed[ys[keep], xs[keep]] = 255

    kernel_size = _scaled_odd_kernel(reconnect_radius, 1.0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    near_seed = cv2.dilate(seed, kernel, iterations=1) > 0

    centerline = np.zeros_like(binary)
    centerline[candidate & near_seed] = 255
    return centerline, skeleton


def extract_pattern_centerlines(
    image_path: str,
    out_line_path: str = None,
    bin_thresh: int = 127,
    processing_scale: float = 0.30,
    min_band_radius: int = 6,
    max_band_radius: int = 80,
    probe_scale: float = 1.35,
    probe_angles: int = 8,
    reconnect_radius: int = 35,
    line_thickness: int = 5,
    draw_roi: bool = False,
    return_result_image: bool = False,
    jpeg_quality: int = 90,
    png_compression: int = 3,
):
    """
    Extract curved centerlines from the black pattern bands.

    Unlike Hough line detection, this keeps the skeleton as a polyline/curve,
    so slightly bent pattern bands stay centered instead of being forced into
    one straight fitted segment.
    """
    if not 0 < processing_scale <= 1:
        raise ValueError("processing_scale must be in the range (0, 1].")
    if probe_angles < 4:
        raise ValueError("probe_angles must be >= 4.")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    if processing_scale < 1.0:
        work_img = cv2.resize(
            img,
            None,
            fx=processing_scale,
            fy=processing_scale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        work_img = img

    _, binary = cv2.threshold(work_img, bin_thresh, 255, cv2.THRESH_BINARY_INV)
    work_centerline, work_skeleton = _centerline_between_white_regions(
        binary=binary,
        min_radius=max(2.0, min_band_radius * processing_scale),
        max_radius=max(3.0, max_band_radius * processing_scale),
        probe_scale=probe_scale,
        probe_angles=probe_angles,
        reconnect_radius=max(3, int(round(reconnect_radius * processing_scale))),
    )

    centerline_mask = _resize_mask_to_original(work_centerline, img.shape)
    component_count = max(0, cv2.connectedComponents(centerline_mask)[0] - 1)
    centerline_pixels = int(np.count_nonzero(centerline_mask))

    vis = None
    if out_line_path or return_result_image:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if draw_roi:
            cv2.polylines(vis, [get_crop_local_polygon()], isClosed=True, color=(255, 200, 0), thickness=2)

        preview_mask = centerline_mask
        if line_thickness > 1:
            kernel_size = _scaled_odd_kernel(line_thickness, 1.0)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            preview_mask = cv2.dilate(centerline_mask, kernel, iterations=1)
        vis[preview_mask > 0] = (0, 0, 255)

    if out_line_path:
        ext = Path(out_line_path).suffix.lower()
        if ext == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]
        elif ext in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        else:
            params = []
        cv2.imwrite(out_line_path, vis, params)

    return {
        "centerline_mask": centerline_mask,
        "work_centerline_mask": work_centerline,
        "work_skeleton": work_skeleton,
        "result_image": vis,
        "component_count": component_count,
        "centerline_pixels": centerline_pixels,
        "processing_scale": processing_scale,
        # Kept for compatibility with the previous Hough-based caller shape.
        "final_lines": [],
    }


def extract_pattern_lines_optimized(*args, **kwargs):
    """Backward-compatible wrapper. The implementation now returns curved centerlines."""
    return extract_pattern_centerlines(*args, **kwargs)


if __name__ == "__main__":
    total_start = time.perf_counter()
    image_path = "./output/pattern/3_matrix_instances_bw.png"

    try:
        result = extract_pattern_centerlines(
            image_path=image_path,
            out_line_path="./output/centerline/3.jpg",
            processing_scale=0.25,
            line_thickness=5,
        )
        total_time = time.perf_counter() - total_start
        print(f"中心线连通块数: {result['component_count']}")
        print(f"中心线像素数: {result['centerline_pixels']}")
        print(f"{total_time}")
    except Exception as e:
        print(f"执行出错: {e}")

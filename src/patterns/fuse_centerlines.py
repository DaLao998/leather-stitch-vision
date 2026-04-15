from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from src.patterns.centerline import extract_pattern_centerlines
except ModuleNotFoundError:
    from centerline import extract_pattern_centerlines


DEFAULT_INPUTS = [
    "output/pattern/1_matrix_instances_bw.png",
    "output/pattern/2_matrix_instances_bw.png",
    "output/pattern/3_matrix_instances_bw.png",
]
DEFAULT_BASE = "output/pattern/1_matrix_instances_bw.png"
DEFAULT_OUTPUT = "output/centerline/steps_on_1_matrix_instances_bw.jpg"

# BGR. These are deliberately pure channels so overlaps are easy to read:
# 1+2 = yellow, 1+3 = magenta, 2+3 = cyan, all three = white.
STEP_COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
]


def odd_kernel(size: int, minimum: int = 3) -> int:
    size = max(minimum, int(size))
    return size + 1 if size % 2 == 0 else size


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray, jpeg_quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    elif ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    else:
        params = []
    if not cv2.imwrite(str(path), image, params):
        raise RuntimeError(f"Cannot write image: {path}")


def estimate_shift(base_mask: np.ndarray, moving_mask: np.ndarray) -> tuple[float, float, float]:
    """Return dx, dy, response for reporting. The default overlay does not align."""
    base = (base_mask > 0).astype(np.float32)
    moving = (moving_mask > 0).astype(np.float32)
    (dx, dy), response = cv2.phaseCorrelate(base, moving)
    return float(dx), float(dy), float(response)


def apply_inverse_shift(mask: np.ndarray, shift: tuple[float, float]) -> np.ndarray:
    dx, dy = shift
    height, width = mask.shape[:2]
    matrix = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)


def smooth_centerline(mask: np.ndarray, smooth_radius: int = 5) -> np.ndarray:
    """Smooth jagged centerline edges but keep a one-pixel skeleton for later drawing."""
    size = odd_kernel(smooth_radius)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    tube = cv2.dilate((mask > 0).astype(np.uint8) * 255, kernel, iterations=1)
    tube = cv2.morphologyEx(tube, cv2.MORPH_CLOSE, kernel, iterations=1)
    tube = cv2.morphologyEx(tube, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return cv2.ximgproc.thinning(tube, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)


def branchpoint_mask(skeleton: np.ndarray) -> np.ndarray:
    binary = (skeleton > 0).astype(np.uint8)
    neighbors = [
        np.roll(binary, (-1, 0), (0, 1)),
        np.roll(binary, (-1, 1), (0, 1)),
        np.roll(binary, (0, 1), (0, 1)),
        np.roll(binary, (1, 1), (0, 1)),
        np.roll(binary, (1, 0), (0, 1)),
        np.roll(binary, (1, -1), (0, 1)),
        np.roll(binary, (0, -1), (0, 1)),
        np.roll(binary, (-1, -1), (0, 1)),
    ]
    sequence = neighbors + [neighbors[0]]
    transitions = np.zeros_like(binary, dtype=np.uint8)
    for previous, current in zip(sequence[:-1], sequence[1:]):
        transitions += ((previous == 0) & (current == 1)).astype(np.uint8)
    return ((binary > 0) & (transitions >= 3)).astype(np.uint8) * 255


def find_crossing_centers(
    skeleton: np.ndarray,
    cluster_radius: int = 75,
    min_area: int = 200,
) -> list[tuple[float, float]]:
    branches = branchpoint_mask(skeleton)
    size = odd_kernel(cluster_radius)
    branches = cv2.dilate(
        branches,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
        iterations=1,
    )

    count, _, stats, centroids = cv2.connectedComponentsWithStats(branches)
    centers = []
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cx, cy = centroids[label]
            centers.append((float(cx), float(cy)))
    return centers


def repair_crossings(
    skeleton: np.ndarray,
    centers: list[tuple[float, float]],
    radius: int = 50,
    annulus_width: int = 28,
) -> np.ndarray:
    """Clean local crossing areas into two thin crossing strokes."""
    if not centers:
        return skeleton

    repaired = skeleton.copy()
    yy, xx = np.indices(skeleton.shape)

    for cx, cy in centers:
        distance = np.hypot(xx - cx, yy - cy)
        disk = distance <= radius
        ring = (distance > radius) & (distance <= radius + annulus_width) & (skeleton > 0)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(ring.astype(np.uint8))
        points = []
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < 3:
                continue
            ys, xs = np.where(labels == label)
            nearest = np.argmin(np.hypot(xs - cx, ys - cy))
            x = float(xs[nearest])
            y = float(ys[nearest])
            angle = float(np.arctan2(y - cy, x - cx))
            area = int(stats[label, cv2.CC_STAT_AREA])
            points.append((x, y, angle, area))

        if len(points) < 4:
            continue

        points = sorted(points, key=lambda item: item[3], reverse=True)[:4]
        points = sorted(points, key=lambda item: item[2])
        pairs = [(points[0], points[2]), (points[1], points[3])]

        repaired[disk] = 0
        for start, end in pairs:
            cv2.line(
                repaired,
                (int(round(start[0])), int(round(start[1]))),
                (int(round(end[0])), int(round(end[1]))),
                255,
                1,
            )

    repaired = cv2.dilate(repaired, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return cv2.ximgproc.thinning(repaired, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)


def process_single_mask(
    mask: np.ndarray,
    smooth_radius: int,
    repair_crossing: bool,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    processed = smooth_centerline(mask, smooth_radius=smooth_radius)
    centers: list[tuple[float, float]] = []
    if repair_crossing:
        centers = find_crossing_centers(processed)
        processed = repair_crossings(processed, centers)
    return processed, centers


def render_step_overlay(
    base_gray: np.ndarray,
    masks: list[np.ndarray],
    line_thickness: int = 2,
) -> np.ndarray:
    preview = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    overlay = np.zeros_like(preview)

    kernel = None
    if line_thickness > 1:
        size = odd_kernel(line_thickness)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    for index, mask in enumerate(masks):
        draw_mask = mask
        if kernel is not None:
            draw_mask = cv2.dilate(mask, kernel, iterations=1)
        active = draw_mask > 0
        color = np.array(STEP_COLORS[index % len(STEP_COLORS)], dtype=np.uint8)
        overlay[active] = np.maximum(overlay[active], color)

    active = np.any(overlay > 0, axis=2)
    preview[active] = overlay[active]
    return preview


def build_step_overlay(
    image_paths: list[Path],
    base_path: Path,
    processing_scale: float = 0.30,
    smooth_radius: int = 5,
    line_thickness: int = 1,
    repair_crossing: bool = True,
    apply_shifts: bool = False,
) -> dict:
    base = load_gray(base_path)

    raw_masks = []
    for path in image_paths:
        result = extract_pattern_centerlines(str(path), processing_scale=processing_scale, line_thickness=1)
        mask = result["centerline_mask"]
        if mask.shape != base.shape:
            raise ValueError(f"Size mismatch: {path} {mask.shape} != {base.shape}")
        raw_masks.append(mask)

    shifts = [(0.0, 0.0, 1.0)]
    for moving in raw_masks[1:]:
        shifts.append(estimate_shift(raw_masks[0], moving))

    processed_masks = []
    crossing_counts = []
    for index, mask in enumerate(raw_masks):
        work_mask = mask
        if apply_shifts and index > 0:
            dx, dy, _ = shifts[index]
            work_mask = apply_inverse_shift(mask, (dx, dy))
        processed, centers = process_single_mask(work_mask, smooth_radius=smooth_radius, repair_crossing=repair_crossing)
        processed_masks.append(processed)
        crossing_counts.append(len(centers))

    preview = render_step_overlay(base, processed_masks, line_thickness=line_thickness)

    return {
        "base": base,
        "preview": preview,
        "masks": processed_masks,
        "shifts": shifts,
        "crossing_counts": crossing_counts,
        "line_pixels": [int(np.count_nonzero(mask)) for mask in processed_masks],
        "component_counts": [max(0, cv2.connectedComponents(mask)[0] - 1) for mask in processed_masks],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw step-by-step pattern centerlines onto image 1.")
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS, help="Pattern matrix images: 1/2/3.")
    parser.add_argument("--base", default=DEFAULT_BASE, help="Base image to draw onto.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output overlay preview path.")
    parser.add_argument("--processing-scale", type=float, default=0.30)
    parser.add_argument("--smooth-radius", type=int, default=5)
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--apply-shifts", action="store_true", help="Apply estimated shifts before drawing. Default is off.")
    parser.add_argument("--no-crossing-repair", action="store_true", help="Disable local crossing cleanup.")
    return parser.parse_args()


def main() -> None:
    start = time.perf_counter()
    args = parse_args()

    image_paths = [Path(path) for path in args.inputs]
    result = build_step_overlay(
        image_paths=image_paths,
        base_path=Path(args.base),
        processing_scale=args.processing_scale,
        smooth_radius=args.smooth_radius,
        line_thickness=args.line_thickness,
        repair_crossing=not args.no_crossing_repair,
        apply_shifts=args.apply_shifts,
    )

    write_image(Path(args.output), result["preview"])

    print("Done. Drawn separately; no fusion into one final line.")
    print("Colors: 1=red, 2=green, 3=blue; overlaps mix by channel.")
    print(f"Apply shifts: {args.apply_shifts}")
    print("Estimated shifts dx, dy, response:")
    for path, (dx, dy, response), crossings, pixels, components in zip(
        image_paths,
        result["shifts"],
        result["crossing_counts"],
        result["line_pixels"],
        result["component_counts"],
    ):
        print(
            f"  {path.name}: dx={dx:.3f}, dy={dy:.3f}, response={response:.3f}, "
            f"crossings={crossings}, components={components}, pixels={pixels}"
        )
    print(f"Output: {args.output}")
    print(f"Elapsed: {time.perf_counter() - start:.3f}s")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.utils import derive_roi_path, iter_mask_paths, load_binary_mask, load_bgr_image, write_image

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:
    cKDTree = None


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

    if len(points) < 32:
        raise RuntimeError("Too few valid hole centers.")
    return np.asarray(points, dtype=np.float32)


def estimate_spacing(points: np.ndarray) -> float:
    if len(points) < 2:
        raise RuntimeError("Not enough points to estimate spacing.")

    if cKDTree is not None:
        dists, _ = cKDTree(points).query(points, k=2)
        nearest = dists[:, 1]
        nearest = nearest[np.isfinite(nearest)]
        if len(nearest) == 0:
            raise RuntimeError("Failed to estimate spacing.")
        return float(np.median(nearest))

    nearest = np.full((len(points),), np.inf, dtype=np.float32)
    block = 512
    for start in range(0, len(points), block):
        end = min(len(points), start + block)
        chunk = points[start:end]
        diff = chunk[:, None, :] - points[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        row_idx = np.arange(end - start)
        dist2[row_idx, start:end] = np.inf
        nearest[start:end] = np.sqrt(np.min(dist2, axis=1))

    nearest = nearest[np.isfinite(nearest)]
    if len(nearest) == 0:
        raise RuntimeError("Failed to estimate spacing.")
    return float(np.median(nearest))


def normalize_angle_deg(angle: float) -> float:
    while angle < -90.0:
        angle += 180.0
    while angle >= 90.0:
        angle -= 180.0
    return angle


@dataclass
class HoleRow:
    angle_deg: float
    normal_coord: float
    points: np.ndarray


def query_point_neighbors(points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 2:
        raise RuntimeError("Not enough points for neighbor query.")

    k = max(1, min(int(k), len(points) - 1))
    if cKDTree is not None:
        dists, indices = cKDTree(points).query(points, k=k + 1)
        return dists[:, 1:], indices[:, 1:]

    dists_out = np.full((len(points), k), np.inf, dtype=np.float32)
    indices_out = np.full((len(points), k), -1, dtype=np.int32)
    block = 256

    for start in range(0, len(points), block):
        end = min(len(points), start + block)
        chunk = points[start:end]
        diff = chunk[:, None, :] - points[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        row_idx = np.arange(end - start)
        dist2[row_idx, start:end] = np.inf

        nearest = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
        nearest_dist2 = np.take_along_axis(dist2, nearest, axis=1)
        order = np.argsort(nearest_dist2, axis=1)
        indices_out[start:end] = np.take_along_axis(nearest, order, axis=1)
        dists_out[start:end] = np.sqrt(np.take_along_axis(nearest_dist2, order, axis=1)).astype(np.float32)

    return dists_out, indices_out


def estimate_row_angles(points: np.ndarray, spacing: float) -> list[float]:
    dists, indices = query_point_neighbors(points, k=6)
    positive: list[float] = []
    negative: list[float] = []
    min_dist = spacing * 0.45
    max_dist = spacing * 1.8

    for idx in range(len(points)):
        for dist, neighbor in zip(dists[idx], indices[idx]):
            if neighbor < 0 or not np.isfinite(dist):
                continue
            if dist < min_dist or dist > max_dist:
                continue

            dx, dy = points[neighbor] - points[idx]
            angle = normalize_angle_deg(float(np.degrees(np.arctan2(dy, dx))))
            if abs(angle) < 20.0:
                continue

            if angle > 0:
                positive.append(angle)
            else:
                negative.append(angle)

    if len(positive) < 8 or len(negative) < 8:
        raise RuntimeError("Failed to estimate dominant hole-row angles.")

    return [float(np.median(negative)), float(np.median(positive))]


def cluster_rows(points: np.ndarray, angle_deg: float, spacing: float) -> list[HoleRow]:
    theta = np.radians(angle_deg)
    direction = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float32)
    normal = np.asarray([-np.sin(theta), np.cos(theta)], dtype=np.float32)
    normal_values = points @ normal
    along_values = points @ direction
    order = np.argsort(normal_values)
    tolerance = max(2.0, spacing * 0.38)

    clusters: list[list[int]] = []
    current: list[int] = [int(order[0])]
    current_sum = float(normal_values[order[0]])

    for raw_idx in order[1:]:
        idx = int(raw_idx)
        center = current_sum / float(len(current))
        if abs(float(normal_values[idx]) - center) <= tolerance:
            current.append(idx)
            current_sum += float(normal_values[idx])
        else:
            clusters.append(current)
            current = [idx]
            current_sum = float(normal_values[idx])
    clusters.append(current)

    rows: list[HoleRow] = []
    for cluster in clusters:
        cluster_idx = np.asarray(cluster, dtype=np.int32)
        row_points = points[cluster_idx]
        row_along = along_values[cluster_idx]
        row_points = row_points[np.argsort(row_along)]
        rows.append(
            HoleRow(
                angle_deg=angle_deg,
                normal_coord=float(np.mean(normal_values[cluster_idx])),
                points=row_points,
            )
        )

    rows.sort(key=lambda row: row.normal_coord)
    return rows


def estimate_row_gap(rows: list[HoleRow], spacing: float) -> float:
    if len(rows) < 2:
        raise RuntimeError("Too few hole rows.")

    coords = np.asarray([row.normal_coord for row in rows], dtype=np.float32)
    diffs = np.diff(coords)
    diffs = diffs[diffs > max(1.0, spacing * 0.20)]
    if len(diffs) == 0:
        return float(spacing)
    return float(np.median(diffs))


def dedupe_rows(rows: list[HoleRow], gap_tolerance: float) -> list[HoleRow]:
    if not rows:
        return []

    rows = sorted(rows, key=lambda row: row.normal_coord)
    deduped: list[HoleRow] = [rows[0]]
    for row in rows[1:]:
        prev = deduped[-1]
        if abs(row.normal_coord - prev.normal_coord) <= gap_tolerance:
            prev_support = (len(prev.points), float(np.linalg.norm(prev.points[-1] - prev.points[0])))
            row_support = (len(row.points), float(np.linalg.norm(row.points[-1] - row.points[0])))
            if row_support > prev_support:
                deduped[-1] = row
        else:
            deduped.append(row)
    return deduped


def refine_row_points(row: HoleRow, spacing: float) -> HoleRow | None:
    if len(row.points) < 6:
        return None

    points = row.points.astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    direction = np.asarray([vx, vy], dtype=np.float32)
    origin = np.asarray([x0, y0], dtype=np.float32)
    offsets = points - origin
    # Use perpendicular distance to reject points that were clustered into the row by accident.
    perp = np.abs(offsets[:, 0] * (-direction[1]) + offsets[:, 1] * direction[0])
    keep = perp <= max(1.6, spacing * 0.16)
    refined = points[keep]
    if len(refined) < 6:
        return None

    projection = refined @ direction
    refined = refined[np.argsort(projection)]
    return HoleRow(angle_deg=row.angle_deg, normal_coord=row.normal_coord, points=refined)


def select_boundary_rows(rows: list[HoleRow], spacing: float) -> list[HoleRow]:
    if len(rows) < 3:
        return []

    refined_rows = [refined for row in rows if (refined := refine_row_points(row, spacing)) is not None]
    if len(refined_rows) < 3:
        return []

    median_count = float(np.median([len(row.points) for row in refined_rows]))
    min_count = max(6.0, median_count * 0.30)
    candidate_rows: list[HoleRow] = []

    for row in refined_rows:
        if len(row.points) < min_count:
            continue

        span = float(np.linalg.norm(row.points[-1] - row.points[0]))
        if span < spacing * 5.0:
            continue

        candidate_rows.append(row)

    if len(candidate_rows) < 3:
        return []

    nominal_gap = estimate_row_gap(candidate_rows, spacing)
    selected: list[HoleRow] = []

    for idx in range(1, len(candidate_rows) - 1):
        row = candidate_rows[idx]
        prev_gap = row.normal_coord - candidate_rows[idx - 1].normal_coord
        next_gap = candidate_rows[idx + 1].normal_coord - row.normal_coord
        gap_large = max(prev_gap, next_gap)
        gap_small = min(prev_gap, next_gap)

        if gap_large >= nominal_gap * 1.60 and gap_small <= nominal_gap * 1.40:
            selected.append(row)

    span_arr = np.asarray([float(np.linalg.norm(row.points[-1] - row.points[0])) for row in refined_rows], dtype=np.float32)
    count_arr = np.asarray([len(row.points) for row in refined_rows], dtype=np.float32)
    for idx in range(2, len(refined_rows) - 2):
        local_span = np.concatenate((span_arr[idx - 2 : idx], span_arr[idx + 1 : idx + 3]))
        local_count = np.concatenate((count_arr[idx - 2 : idx], count_arr[idx + 1 : idx + 3]))
        span_ref = float(np.median(local_span))
        count_ref = float(np.median(local_count))
        if span_ref <= 0.0 or count_ref <= 0.0:
            continue

        # Some samples keep a faint, partially broken row inside a real gap.
        # Use that local support valley to recover the two boundary rows around it.
        if span_arr[idx] >= span_ref * 0.86 or count_arr[idx] >= count_ref * 0.86:
            continue

        left = refined_rows[idx - 1]
        right = refined_rows[idx + 1]
        left_span = float(np.linalg.norm(left.points[-1] - left.points[0]))
        right_span = float(np.linalg.norm(right.points[-1] - right.points[0]))
        if left_span < span_ref * 0.92 or right_span < span_ref * 0.92:
            continue
        if len(left.points) < count_ref * 0.92 or len(right.points) < count_ref * 0.92:
            continue

        selected.append(left)
        selected.append(right)

    return dedupe_rows(selected, gap_tolerance=nominal_gap * 0.45)


def draw_center_points(shape: tuple[int, int], points: np.ndarray, radius: int = 1) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for point in points:
        cv2.circle(mask, (int(round(point[0])), int(round(point[1]))), radius, 255, -1)
    return mask


def draw_boundary_centerlines(
    shape: tuple[int, int],
    rows: list[HoleRow],
    spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    centers_mask = np.zeros(shape, dtype=np.uint8)
    line_mask = np.zeros(shape, dtype=np.uint8)

    for row in rows:
        for point in row.points:
            cv2.circle(centers_mask, (int(round(point[0])), int(round(point[1]))), 1, 255, -1)

        for p1, p2 in zip(row.points[:-1], row.points[1:]):
            if float(np.linalg.norm(p2 - p1)) > spacing * 2.25:
                continue
            cv2.line(
                line_mask,
                (int(round(p1[0])), int(round(p1[1]))),
                (int(round(p2[0])), int(round(p2[1]))),
                255,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    return centers_mask, line_mask


def overlay_centerline(image: np.ndarray, centers_mask: np.ndarray, line_mask: np.ndarray) -> np.ndarray:
    preview = image.copy()
    points_thick = cv2.dilate(centers_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    lines_thick = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    preview[points_thick > 0] = (0, 255, 255)
    preview[lines_thick > 0] = (0, 0, 255)
    return preview


def build_pattern_artifacts(
    hole_mask: np.ndarray,
    roi_image: np.ndarray,
    max_dim: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    del max_dim

    centers = extract_hole_centers(hole_mask)
    spacing = estimate_spacing(centers)
    angles = estimate_row_angles(centers, spacing)

    all_centers_mask = draw_center_points(hole_mask.shape, centers, radius=1)
    boundary_centers_mask = np.zeros_like(all_centers_mask)
    line_mask = np.zeros_like(all_centers_mask)

    for angle_deg in angles:
        rows = cluster_rows(centers, angle_deg=angle_deg, spacing=spacing)
        boundary_rows = select_boundary_rows(rows, spacing=spacing)
        family_centers_mask, family_line_mask = draw_boundary_centerlines(hole_mask.shape, boundary_rows, spacing=spacing)
        boundary_centers_mask = cv2.bitwise_or(boundary_centers_mask, family_centers_mask)
        line_mask = cv2.bitwise_or(line_mask, family_line_mask)

    preview = overlay_centerline(roi_image, boundary_centers_mask, line_mask)
    return all_centers_mask, boundary_centers_mask, line_mask, preview


def process_path(hole_path: Path, roi_path: Path, output_dir: Path, max_dim: int) -> None:
    hole_mask = load_binary_mask(hole_path)
    roi_image = load_bgr_image(roi_path)
    band, skeleton, line_mask, preview = build_pattern_artifacts(hole_mask, roi_image, max_dim=max_dim)

    stem = hole_path.stem.replace("_holes_bw", "")
    write_image(output_dir / f"{stem}_pattern_band.png", band)
    write_image(output_dir / f"{stem}_pattern_skeleton.png", skeleton)
    write_image(output_dir / f"{stem}_pattern_centerline.png", line_mask)
    write_image(output_dir / f"{stem}_pattern_preview.png", preview)


def main() -> None:
    start = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Extract boundary hole centerlines directly from perforation center rows."
    )
    parser.add_argument("--holes", default="output/holes", help="Hole mask file or directory.")
    parser.add_argument("--roi", default="output/roi", help="ROI image file or directory.")
    parser.add_argument("--output", default="output/pattern", help="Directory for pattern outputs.")
    parser.add_argument("--max-dim", type=int, default=1000, help="Processing size for the long side.")
    args = parser.parse_args()

    hole_paths = iter_mask_paths(Path(args.holes))
    if not hole_paths:
        raise RuntimeError(f"No hole masks found in: {args.holes}")

    roi_input = Path(args.roi)
    output_dir = Path(args.output)
    for hole_path in hole_paths:
        roi_path = roi_input if roi_input.is_file() else derive_roi_path(hole_path, roi_input)
        process_path(hole_path, roi_path, output_dir, max_dim=args.max_dim)

    print(f"pattern: {time.perf_counter() - start:.3f}s")


if __name__ == "__main__":
    main()

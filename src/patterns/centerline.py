from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ==============================================================================
# Global Parameters
# ==============================================================================

DEFAULT_INPUT = "1"
DEFAULT_PATTERN_DIR = Path("output/pattern")
DEFAULT_OUTPUT = "output/centerline"

PROCESSING_SCALE = 0.25      # 处理时的降采样比例；先缩小到原图的 25% 再做中心线提取，用于降低骨架化和距离变换的计算量
MIN_BAND_RADIUS = 6          # 花样带允许的最小半宽（像素）；距离变换值小于该值的骨架点会被视为过细噪声或无效中心线
MAX_BAND_RADIUS = 80         # 花样带允许的最大半宽（像素）；距离变换值大于该值的骨架点通常来自大块区域或外边框，会被过滤
PROBE_SCALE = 1.35           # 两侧白区探测半径相对距离变换值的放大系数；探测半径 = 当前局部半宽 × 1.35
PROBE_ANGLES = 8             # 两侧白区判定时的探测方向数；越大方向覆盖越密，但计算量也越高
RECONNECT_RADIUS = 35        # 中心线重连半径（像素）；对有效种子点做局部膨胀，用于恢复交叉口附近可能断掉的骨架点

INNER_INSET = 28             # 内部有效矩形向内收缩的边距（像素）；用于避开图像外围边框和边缘干扰区域
HOUGH_THRESHOLD = 80         # 概率霍夫变换的累积阈值；越大越严格，只保留证据更充分的直线段
MIN_LINE_LENGTH = 160        # 霍夫检测允许的最短线段长度（像素）；短于该值的候选线段会被忽略
MAX_LINE_GAP = 80            # 霍夫检测中同一直线段内部允许连接的最大断裂间隔（像素）
RHO_CLUSTER = 55.0           # 平行线聚类时按法向距离分组的阈值（像素）；用于区分“同方向但不同位置”的多条直线
EXTENT_PAD = 120.0           # 交点和边界点判定时，对拟合直线有效范围向两端放宽的容差（像素）
MIN_SEGMENT_LENGTH = 80.0    # 有效几何线段的最小长度（像素）；短于该值的节点间连线通常被视为无效
MIN_BLACK_RATIO = 0.45       # 候选线段沿线采样时落在黑色花样区域中的最小占比；低于该值说明该线段与真实花样不够吻合
LINE_THICKNESS = 2           # 结果可视化时绘制线段的线宽（像素）


# ==============================================================================


class StageTimer:
    def __init__(self) -> None:
        self.records = defaultdict(float)

    def add(self, key: str, elapsed: float) -> None:
        self.records[key] += elapsed

    def measure(self, key: str):
        return _TimerContext(self, key)

    def as_dict(self) -> dict[str, float]:
        return dict(self.records)


class _TimerContext:
    def __init__(self, timer: StageTimer, key: str) -> None:
        self.timer = timer
        self.key = key
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.add(self.key, time.perf_counter() - self.start)


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


def _ratio_0_1(value: str) -> float:
    fvalue = float(value)
    if not (0.0 <= fvalue <= 1.0):
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return fvalue


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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_pattern_path(value: str) -> Path:
    path = Path(value)
    if path.exists() or path.suffix:
        return path
    if value.isdigit():
        return DEFAULT_PATTERN_DIR / f"{value}_matrix_instances_bw.png"
    return DEFAULT_PATTERN_DIR / f"{value}_matrix_instances_bw.png"


def iter_pattern_paths(input_value: str) -> list[Path]:
    input_path = Path(input_value)
    if input_value.isdigit() or input_path.is_file() or input_path.suffix:
        return [resolve_pattern_path(input_value)]
    if input_path.is_dir():
        return sorted(input_path.glob("*_matrix_instances_bw.png"))
    return [resolve_pattern_path(input_value)]


def output_stem(image_path: Path) -> str:
    stem = image_path.stem
    if stem.endswith("_matrix_instances_bw"):
        stem = stem[: -len("_matrix_instances_bw")]
    return f"geo_{stem}"


def resolve_output_paths(image_path: Path, output: str) -> tuple[Path, Path]:
    output_path = Path(output)
    if output_path.suffix:
        image_output = output_path
        json_output = output_path.with_suffix(".json")
        return image_output, json_output

    stem = output_stem(image_path)
    return output_path / f"{stem}.jpg", output_path / f"{stem}.json"


def scaled_odd_kernel(value: int | float, scale: float, minimum: int = 3) -> int:
    size = max(minimum, int(round(float(value) * scale)))
    return size + 1 if size % 2 == 0 else size


def resize_mask_to_shape(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    height, width = shape[:2]
    if mask.shape == (height, width):
        return mask
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def centerline_between_white_regions(
    binary: np.ndarray,
    min_radius: float,
    max_radius: float,
    probe_scale: float,
    probe_angles: int,
    reconnect_radius: int,
) -> np.ndarray:
    skeleton = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    candidate = (skeleton > 0) & (dist >= min_radius) & (dist <= max_radius)
    ys, xs = np.where(candidate)
    if len(xs) == 0:
        return np.zeros_like(binary)

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

    kernel_size = scaled_odd_kernel(reconnect_radius, 1.0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    near_seed = cv2.dilate(seed, kernel, iterations=1) > 0

    centerline = np.zeros_like(binary)
    centerline[candidate & near_seed] = 255
    return centerline


def build_centerline_mask(
    image: np.ndarray,
    processing_scale: float,
    min_band_radius: int,
    max_band_radius: int,
    probe_scale: float,
    probe_angles: int,
    reconnect_radius: int,
) -> np.ndarray:
    if not 0 < processing_scale <= 1:
        raise ValueError("processing_scale must be in the range (0, 1].")
    if probe_angles < 4:
        raise ValueError("probe_angles must be >= 4.")

    if processing_scale < 1.0:
        work_img = cv2.resize(
            image,
            None,
            fx=processing_scale,
            fy=processing_scale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        work_img = image

    _, binary = cv2.threshold(work_img, 127, 255, cv2.THRESH_BINARY_INV)
    work_centerline = centerline_between_white_regions(
        binary=binary,
        min_radius=max(2.0, min_band_radius * processing_scale),
        max_radius=max(3.0, max_band_radius * processing_scale),
        probe_scale=probe_scale,
        probe_angles=probe_angles,
        reconnect_radius=max(3, int(round(reconnect_radius * processing_scale))),
    )
    return resize_mask_to_shape(work_centerline, image.shape)


def estimate_inner_rect(
    image: np.ndarray,
    white_threshold: int = 127,
    min_white_ratio: float = 0.02,
    inset: int = INNER_INSET,
) -> tuple[int, int, int, int]:
    h, w = image.shape[:2]
    white = image > white_threshold
    rows = np.where(np.count_nonzero(white, axis=1) >= int(w * min_white_ratio))[0]
    cols = np.where(np.count_nonzero(white, axis=0) >= int(h * min_white_ratio))[0]
    if rows.size == 0 or cols.size == 0:
        return 0, 0, w - 1, h - 1

    left = int(np.clip(cols[0] + inset, 0, w - 1))
    right = int(np.clip(cols[-1] - inset, 0, w - 1))
    top = int(np.clip(rows[0] + inset, 0, h - 1))
    bottom = int(np.clip(rows[-1] - inset, 0, h - 1))
    if left >= right or top >= bottom:
        return 0, 0, w - 1, h - 1
    return left, top, right, bottom


def angle_vector(angle: float) -> np.ndarray:
    return np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)


def fit_line(points: np.ndarray, direction_hint: np.ndarray | None = None) -> dict:
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    direction = np.array([float(vx), float(vy)], dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    if direction_hint is not None and float(np.dot(direction, direction_hint)) < 0:
        direction *= -1

    origin = np.array([float(x0), float(y0)], dtype=np.float64)
    ts = (points.astype(np.float64) - origin).dot(direction)
    return {
        "direction": direction,
        "origin": origin,
        "tmin": float(ts.min()),
        "tmax": float(ts.max()),
    }


def intersect_lines(line_a: dict, line_b: dict) -> tuple[np.ndarray, float, float] | None:
    da = line_a["direction"]
    db = line_b["direction"]
    oa = line_a["origin"]
    ob = line_b["origin"]
    matrix = np.array([[da[0], -db[0]], [da[1], -db[1]]], dtype=np.float64)
    if abs(float(np.linalg.det(matrix))) < 1e-6:
        return None
    ta, tb = np.linalg.solve(matrix, ob - oa)
    return oa + da * ta, float(ta), float(tb)


def line_rect_intersections(line: dict, rect: tuple[int, int, int, int]) -> list[tuple[np.ndarray, float]]:
    left, top, right, bottom = rect
    direction = line["direction"]
    origin = line["origin"]
    vx, vy = direction
    x0, y0 = origin
    points: list[tuple[np.ndarray, float]] = []

    if abs(vx) > 1e-6:
        for x in (left, right):
            t = (float(x) - x0) / vx
            y = y0 + vy * t
            if top - 1 <= y <= bottom + 1:
                points.append((np.array([float(x), y], dtype=np.float64), float(t)))

    if abs(vy) > 1e-6:
        for y in (top, bottom):
            t = (float(y) - y0) / vy
            x = x0 + vx * t
            if left - 1 <= x <= right + 1:
                points.append((np.array([x, float(y)], dtype=np.float64), float(t)))

    unique: list[tuple[np.ndarray, float]] = []
    for point, t in points:
        if not any(float(np.linalg.norm(point - old_point)) < 2.0 for old_point, _ in unique):
            unique.append((point, t))
    return unique


def point_in_rect(point: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    left, top, right, bottom = rect
    x, y = point
    return left <= x <= right and top <= y <= bottom


def cluster_hough_segments(
    centerline_mask: np.ndarray,
    hough_threshold: int,
    min_line_length: int,
    max_line_gap: int,
    rho_cluster: float,
) -> list[dict]:
    hough = cv2.HoughLinesP(
        centerline_mask,
        1,
        np.pi / 180.0,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if hough is None:
        return []

    segments = []
    for x1, y1, x2, y2 in hough.reshape(-1, 4):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(math.hypot(dx, dy))
        if length < min_line_length:
            continue
        angle = math.atan2(dy, dx) % math.pi
        segments.append(
            {
                "angle": angle,
                "length": length,
                "points": np.array([[x1, y1], [x2, y2]], dtype=np.float64),
            }
        )

    if len(segments) < 2:
        return []

    angle_features = np.array(
        [[math.cos(2.0 * item["angle"]), math.sin(2.0 * item["angle"])] for item in segments],
        dtype=np.float64,
    )
    first = 0
    second = int(np.argmax(np.sum((angle_features - angle_features[first]) ** 2, axis=1)))
    centers = angle_features[[first, second]].copy()
    labels = np.zeros(len(segments), dtype=np.int32)

    for _ in range(20):
        dist = np.sum((angle_features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dist, axis=1)
        for label in (0, 1):
            active = labels == label
            if not np.any(active):
                continue
            center = angle_features[active].mean(axis=0)
            norm = float(np.linalg.norm(center))
            if norm > 1e-9:
                centers[label] = center / norm

    lines: list[dict] = []
    for family in (0, 1):
        family_segments = [segments[i] for i in range(len(segments)) if labels[i] == family]
        if not family_segments:
            continue

        family_angle = 0.5 * math.atan2(float(centers[family, 1]), float(centers[family, 0])) % math.pi
        direction_hint = angle_vector(family_angle)
        normal = np.array([-direction_hint[1], direction_hint[0]], dtype=np.float64)
        entries = [(float(item["points"][0].dot(normal)), item) for item in family_segments]
        entries.sort(key=lambda item: item[0])

        clusters: list[list[tuple[float, dict]]] = []
        for entry in entries:
            if not clusters:
                clusters.append([entry])
                continue
            previous = clusters[-1]
            previous_rho = float(
                np.average([rho for rho, _ in previous], weights=[item["length"] for _, item in previous])
            )
            if abs(entry[0] - previous_rho) > rho_cluster:
                clusters.append([entry])
            else:
                clusters[-1].append(entry)

        for cluster in clusters:
            points = np.vstack([item["points"] for _, item in cluster])
            line = fit_line(points, direction_hint=direction_hint)
            line["family"] = family
            line["support_count"] = int(len(cluster))
            line["support_length"] = round(float(sum(item["length"] for _, item in cluster)), 2)
            lines.append(line)

    return lines


def sample_black_ratio(
    black_mask: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    sample_step: float = 6.0,
) -> float:
    length = float(np.linalg.norm(p1 - p0))
    if length < 1e-6:
        return 0.0
    count = max(2, int(math.ceil(length / sample_step)))
    xs = np.linspace(p0[0], p1[0], count)
    ys = np.linspace(p0[1], p1[1], count)
    xi = np.clip(np.rint(xs).astype(np.int32), 0, black_mask.shape[1] - 1)
    yi = np.clip(np.rint(ys).astype(np.int32), 0, black_mask.shape[0] - 1)
    return float(np.count_nonzero(black_mask[yi, xi])) / float(count)


def detect_geometric_segments(
    image: np.ndarray,
    centerline_mask: np.ndarray,
    inner_rect: tuple[int, int, int, int],
    args: argparse.Namespace,
) -> dict:
    black_mask = image < 128
    lines = cluster_hough_segments(
        centerline_mask=centerline_mask,
        hough_threshold=args.hough_threshold,
        min_line_length=args.min_line_length,
        max_line_gap=args.max_line_gap,
        rho_cluster=args.rho_cluster,
    )

    crossings = []
    for i, line_a in enumerate(lines):
        for j, line_b in enumerate(lines):
            if i >= j or line_a["family"] == line_b["family"]:
                continue
            result = intersect_lines(line_a, line_b)
            if result is None:
                continue
            point, ta, tb = result
            if not point_in_rect(point, inner_rect):
                continue
            if not (line_a["tmin"] - args.extent_pad <= ta <= line_a["tmax"] + args.extent_pad):
                continue
            if not (line_b["tmin"] - args.extent_pad <= tb <= line_b["tmax"] + args.extent_pad):
                continue

            x = int(round(point[0]))
            y = int(round(point[1]))
            if 0 <= y < black_mask.shape[0] and 0 <= x < black_mask.shape[1] and not black_mask[y, x]:
                continue

            crossings.append(
                {
                    "id": f"C{len(crossings)}",
                    "point": point,
                    "line_indices": [i, j],
                    "line_t": {i: ta, j: tb},
                }
            )

    nodes = []
    line_nodes: dict[int, list[dict]] = {index: [] for index in range(len(lines))}
    for crossing in crossings:
        nodes.append({"id": crossing["id"], "kind": "crossing", "point": crossing["point"]})
        for line_index in crossing["line_indices"]:
            line_nodes[line_index].append(
                {
                    "node_id": crossing["id"],
                    "kind": "crossing",
                    "point": crossing["point"],
                    "t": float(crossing["line_t"][line_index]),
                }
            )

    for line_index, line in enumerate(lines):
        for point, t in line_rect_intersections(line, inner_rect):
            existing_items = line_nodes[line_index]
            if existing_items:
                nearest = min(existing_items, key=lambda item: abs(float(item["t"]) - float(t)))
                length_to_nearest = float(np.linalg.norm(point - nearest["point"]))
                if length_to_nearest < 10.0:
                    continue
                if sample_black_ratio(black_mask, point, nearest["point"]) < args.min_black_ratio:
                    continue
            elif not (line["tmin"] - args.extent_pad <= t <= line["tmax"] + args.extent_pad):
                continue

            node_id = f"B{len(nodes)}"
            nodes.append({"id": node_id, "kind": "boundary", "point": point, "line_index": line_index})
            line_nodes[line_index].append(
                {
                    "node_id": node_id,
                    "kind": "boundary",
                    "point": point,
                    "t": float(t),
                }
            )

    segments = []
    seen: set[tuple[str, str]] = set()
    for line_index, items in line_nodes.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda item: item["t"])
        for start, end in zip(items[:-1], items[1:]):
            p0 = start["point"]
            p1 = end["point"]
            length = float(np.linalg.norm(p1 - p0))
            min_length = args.min_segment_length
            if start["kind"] == "boundary" or end["kind"] == "boundary":
                min_length = min(35.0, args.min_segment_length)
            if length < min_length:
                continue
            black_ratio = sample_black_ratio(black_mask, p0, p1)
            if black_ratio < args.min_black_ratio:
                continue
            key = tuple(sorted((start["node_id"], end["node_id"])))
            if key in seen:
                continue
            seen.add(key)
            segments.append(
                {
                    "id": f"S{len(segments)}",
                    "line_index": line_index,
                    "from": start["node_id"],
                    "to": end["node_id"],
                    "length": round(length, 2),
                    "black_ratio": round(black_ratio, 3),
                    "points": [
                        [int(round(p0[0])), int(round(p0[1]))],
                        [int(round(p1[0])), int(round(p1[1]))],
                    ],
                }
            )

    serializable_lines = []
    for index, line in enumerate(lines):
        origin = line["origin"]
        direction = line["direction"]
        serializable_lines.append(
            {
                "id": index,
                "family": int(line["family"]),
                "origin": [round(float(origin[0]), 3), round(float(origin[1]), 3)],
                "direction": [round(float(direction[0]), 6), round(float(direction[1]), 6)],
                "support_count": int(line["support_count"]),
                "support_length": float(line["support_length"]),
            }
        )

    serializable_nodes = []
    for node in nodes:
        point = node["point"]
        item = {
            "id": node["id"],
            "kind": node["kind"],
            "point": [int(round(point[0])), int(round(point[1]))],
        }
        if "line_index" in node:
            item["line_index"] = int(node["line_index"])
        serializable_nodes.append(item)

    return {
        "lines": serializable_lines,
        "nodes": serializable_nodes,
        "segments": segments,
    }


def render_result(
    image: np.ndarray,
    inner_rect: tuple[int, int, int, int],
    nodes: list[dict],
    segments: list[dict],
    line_thickness: int,
    draw_inner_rect: bool,
) -> np.ndarray:
    preview = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if draw_inner_rect:
        left, top, right, bottom = inner_rect
        cv2.rectangle(preview, (left, top), (right, bottom), (0, 180, 255), 1)

    for segment in segments:
        p0, p1 = segment["points"]
        cv2.line(preview, tuple(p0), tuple(p1), (0, 0, 255), line_thickness, lineType=cv2.LINE_AA)

    for node in nodes:
        point = tuple(node["point"])
        color = (0, 255, 0) if node["kind"] == "crossing" else (255, 255, 0)
        cv2.circle(preview, point, max(3, line_thickness + 2), color, -1, lineType=cv2.LINE_AA)

    return preview


def process_path(image_path: Path, output_dir_or_file: str, args: argparse.Namespace) -> tuple[float, float, float]:
    timer = StageTimer()

    with timer.measure("load_image"):
        image = load_gray(image_path)

    with timer.measure("estimate_inner_rect"):
        inner_rect = estimate_inner_rect(image, inset=args.inner_inset)

    with timer.measure("extract_centerline"):
        centerline_mask = build_centerline_mask(
            image=image,
            processing_scale=args.processing_scale,
            min_band_radius=args.min_band_radius,
            max_band_radius=args.max_band_radius,
            probe_scale=args.probe_scale,
            probe_angles=args.probe_angles,
            reconnect_radius=args.reconnect_radius,
        )

    with timer.measure("detect_segments"):
        geometry = detect_geometric_segments(
            image=image,
            centerline_mask=centerline_mask,
            inner_rect=inner_rect,
            args=args,
        )

    with timer.measure("render"):
        preview = render_result(
            image=image,
            inner_rect=inner_rect,
            nodes=geometry["nodes"],
            segments=geometry["segments"],
            line_thickness=args.line_thickness,
            draw_inner_rect=args.draw_inner_rect,
        )

    image_output, json_output = resolve_output_paths(image_path, output_dir_or_file)
    with timer.measure("write_image"):
        write_image(image_output, preview)

    payload = {
        "source": str(image_path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "inner_rect": list(map(int, inner_rect)),
        "line_count": len(geometry["lines"]),
        "node_count": len(geometry["nodes"]),
        "segment_count": len(geometry["segments"]),
        **geometry,
    }
    with timer.measure("write_json"):
        write_json(json_output, payload)

    t = timer.as_dict()
    stage1 = t.get("load_image", 0.0) + t.get("estimate_inner_rect", 0.0) + t.get("extract_centerline", 0.0)
    stage2 = t.get("detect_segments", 0.0)
    stage3 = t.get("render", 0.0) + t.get("write_image", 0.0) + t.get("write_json", 0.0)

    print(
        f"{image_path.name}: lines={payload['line_count']}, nodes={payload['node_count']}, "
        f"segments={payload['segment_count']}"
    )
    print(f"  output: {image_output}")
    print(f"  json:   {json_output}")
    for key, elapsed in sorted(t.items(), key=lambda item: item[1], reverse=True):
        print(f"  {key:<20} {elapsed:.4f}s")

    return stage1, stage2, stage3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract geometric pattern centerlines from *_matrix_instances_bw.png.")
    parser.add_argument("image", nargs="?", help="Image id/path. Example: 1 or output/pattern/1_matrix_instances_bw.png.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Image id/path or directory. Default: 1.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory, or explicit image path for one input.")

    group_center = parser.add_argument_group("Centerline Extraction Parameters")
    group_center.add_argument("--processing-scale", type=_positive_float, default=PROCESSING_SCALE)
    group_center.add_argument("--min-band-radius", type=_positive_int, default=MIN_BAND_RADIUS)
    group_center.add_argument("--max-band-radius", type=_positive_int, default=MAX_BAND_RADIUS)
    group_center.add_argument("--probe-scale", type=_positive_float, default=PROBE_SCALE)
    group_center.add_argument("--probe-angles", type=_positive_int, default=PROBE_ANGLES)
    group_center.add_argument("--reconnect-radius", type=_positive_int, default=RECONNECT_RADIUS)

    group_geo = parser.add_argument_group("Geometric Segment Parameters")
    group_geo.add_argument("--inner-inset", type=int, default=INNER_INSET)
    group_geo.add_argument("--hough-threshold", type=_positive_int, default=HOUGH_THRESHOLD)
    group_geo.add_argument("--min-line-length", type=_positive_int, default=MIN_LINE_LENGTH)
    group_geo.add_argument("--max-line-gap", type=_positive_int, default=MAX_LINE_GAP)
    group_geo.add_argument("--rho-cluster", type=_positive_float, default=RHO_CLUSTER)
    group_geo.add_argument("--extent-pad", type=_positive_float, default=EXTENT_PAD)
    group_geo.add_argument("--min-segment-length", type=_positive_float, default=MIN_SEGMENT_LENGTH)
    group_geo.add_argument("--min-black-ratio", type=_ratio_0_1, default=MIN_BLACK_RATIO)

    group_output = parser.add_argument_group("Output Parameters")
    group_output.add_argument("--line-thickness", type=_positive_int, default=LINE_THICKNESS)
    group_output.add_argument("--draw-inner-rect", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.min_band_radius >= args.max_band_radius:
        raise ValueError("min_band_radius must be smaller than max_band_radius")
    if args.probe_angles < 4:
        raise ValueError("probe_angles must be >= 4")


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    validate_args(args)

    input_value = args.image or args.input
    image_paths = iter_pattern_paths(input_value)
    if not image_paths:
        raise RuntimeError(f"No pattern images found under: {input_value}")

    total1, total2, total3 = 0.0, 0.0, 0.0
    for image_path in image_paths:
        e1, e2, e3 = process_path(image_path, args.output, args)
        total1 += e1
        total2 += e2
        total3 += e3

    total = total1 + total2 + total3
    total_time = time.perf_counter() - total_start
    print(f"time1: {total1:.3f}s + time2: {total2:.3f}s + time3: {total3:.3f}s = algorithm: {total:.3f}s")
    print(f"elapsed total: {total_time:.3f}s")


if __name__ == "__main__":
    main()

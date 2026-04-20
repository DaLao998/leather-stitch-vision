from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class Node:
    node_id: str
    kind: str
    point: Tuple[float, float]


@dataclass
class Segment:
    seg_id: str
    from_node: str
    to_node: str
    p0: Tuple[float, float]
    p1: Tuple[float, float]
    length: float
    midpoint: Tuple[float, float]
    angle_rad: float
    angle_deg: float

    @staticmethod
    def from_json_item(item: dict) -> "Segment":
        p0 = tuple(map(float, item["points"][0]))
        p1 = tuple(map(float, item["points"][1]))

        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        length = math.hypot(dx, dy)
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        midpoint = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)

        return Segment(
            seg_id=str(item["id"]),
            from_node=str(item["from"]),
            to_node=str(item["to"]),
            p0=p0,
            p1=p1,
            length=length,
            midpoint=midpoint,
            angle_rad=angle_rad,
            angle_deg=angle_deg,
        )


@dataclass
class GeoData:
    source: str
    width: int
    height: int
    inner_rect: Tuple[int, int, int, int]
    nodes: List[Node]
    segments: List[Segment]


@dataclass
class SegmentMatchResult:
    ref_segment: str
    target_segment: str
    ref_from_node: str
    ref_to_node: str
    target_from_node: str
    target_to_node: str

    rotation_deg: float
    translation_midpoint_xy: Tuple[float, float]
    rigid_translation_xy: Tuple[float, float]

    length_ref: float
    length_target: float
    length_ratio: float

    midpoint_ref: Tuple[float, float]
    midpoint_target: Tuple[float, float]
    angle_ref_deg: float
    angle_target_deg: float


# ==============================================================================
# Basic Utils
# ==============================================================================

def normalize_angle_diff_deg(diff_deg: float) -> float:
    while diff_deg <= -180.0:
        diff_deg += 360.0
    while diff_deg > 180.0:
        diff_deg -= 360.0
    return diff_deg


def rotation_matrix(theta_rad: float) -> np.ndarray:
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def vec_sub(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_to_np(p: Tuple[float, float]) -> np.ndarray:
    return np.array([p[0], p[1]], dtype=np.float64)


def round_point(p: Tuple[float, float], ndigits: int = 6) -> Tuple[float, float]:
    return (round(float(p[0]), ndigits), round(float(p[1]), ndigits))


# ==============================================================================
# File IO
# ==============================================================================

def load_geo_json(path: Path) -> GeoData:
    payload = json.loads(path.read_text(encoding="utf-8"))

    nodes = [
        Node(
            node_id=str(item["id"]),
            kind=str(item["kind"]),
            point=(float(item["point"][0]), float(item["point"][1])),
        )
        for item in payload.get("nodes", [])
    ]

    segments = [Segment.from_json_item(item) for item in payload.get("segments", [])]

    return GeoData(
        source=str(payload.get("source", path)),
        width=int(payload["width"]),
        height=int(payload["height"]),
        inner_rect=tuple(map(int, payload.get("inner_rect", [0, 0, 0, 0]))),
        nodes=nodes,
        segments=segments,
    )


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ==============================================================================
# Hungarian Algorithm (no scipy dependency)
# ==============================================================================

def hungarian_algorithm(cost: np.ndarray) -> List[Tuple[int, int]]:
    """
    Solve rectangular assignment using DP over subsets.
    适合你这种点数不大的情况，避免依赖 scipy。
    返回 [(row_idx, col_idx), ...]
    """
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return []

    # 让 rows <= cols，便于子集DP
    transposed = False
    work = cost
    if n_rows > n_cols:
        work = cost.T.copy()
        n_rows, n_cols = work.shape
        transposed = True

    max_mask = 1 << n_cols
    dp = [float("inf")] * max_mask
    parent: List[Optional[Tuple[int, int]]] = [None] * max_mask
    dp[0] = 0.0

    for mask in range(max_mask):
        i = bin(mask).count("1")
        if i >= n_rows:
            continue
        base = dp[mask]
        if math.isinf(base):
            continue
        for j in range(n_cols):
            if mask & (1 << j):
                continue
            new_mask = mask | (1 << j)
            value = base + float(work[i, j])
            if value < dp[new_mask]:
                dp[new_mask] = value
                parent[new_mask] = (mask, j)

    best_mask = None
    best_value = float("inf")
    for mask in range(max_mask):
        if bin(mask).count("1") == n_rows and dp[mask] < best_value:
            best_value = dp[mask]
            best_mask = mask

    if best_mask is None:
        return []

    assignment_rev = []
    mask = best_mask
    i = n_rows - 1
    while i >= 0:
        prev = parent[mask]
        if prev is None:
            break
        prev_mask, j = prev
        assignment_rev.append((i, j))
        mask = prev_mask
        i -= 1

    assignment = list(reversed(assignment_rev))

    if transposed:
        # 原来是 cost.T，所以现在要交换回来
        return [(j, i) for i, j in assignment]
    return assignment


# ==============================================================================
# Point Matching
# ==============================================================================

def build_node_degree_map(segments: List[Segment]) -> Dict[str, int]:
    degree: Dict[str, int] = {}
    for seg in segments:
        degree[seg.from_node] = degree.get(seg.from_node, 0) + 1
        degree[seg.to_node] = degree.get(seg.to_node, 0) + 1
    return degree


def estimate_global_translation(
    ref_nodes: List[Node],
    tgt_nodes: List[Node],
) -> Tuple[float, float]:
    """
    粗估一个全局平移：按同类点的中心差来估计。
    """
    if not ref_nodes or not tgt_nodes:
        return (0.0, 0.0)

    ref_center = np.mean(np.array([n.point for n in ref_nodes], dtype=np.float64), axis=0)
    tgt_center = np.mean(np.array([n.point for n in tgt_nodes], dtype=np.float64), axis=0)
    delta = tgt_center - ref_center
    return (float(delta[0]), float(delta[1]))


def build_node_cost_matrix(
    ref_nodes: List[Node],
    tgt_nodes: List[Node],
    ref_degree: Dict[str, int],
    tgt_degree: Dict[str, int],
    global_translation: Tuple[float, float],
    degree_penalty: float = 20.0,
) -> np.ndarray:
    """
    代价主要看：
    1. 点位置距离（考虑全局平移）
    2. 度数差惩罚
    """
    n_ref = len(ref_nodes)
    n_tgt = len(tgt_nodes)
    cost = np.zeros((n_ref, n_tgt), dtype=np.float64)

    tx, ty = global_translation

    for i, rn in enumerate(ref_nodes):
        ref_shifted = (rn.point[0] + tx, rn.point[1] + ty)
        rd = ref_degree.get(rn.node_id, 0)
        for j, tn in enumerate(tgt_nodes):
            td = tgt_degree.get(tn.node_id, 0)
            dist = euclidean(ref_shifted, tn.point)
            deg_cost = abs(rd - td) * degree_penalty
            cost[i, j] = dist + deg_cost

    return cost


def match_nodes_by_kind(
    ref_geo: GeoData,
    tgt_geo: GeoData,
) -> Dict[str, str]:
    ref_degree = build_node_degree_map(ref_geo.segments)
    tgt_degree = build_node_degree_map(tgt_geo.segments)

    mapping: Dict[str, str] = {}

    for kind in ["crossing", "boundary"]:
        ref_nodes = [n for n in ref_geo.nodes if n.kind == kind]
        tgt_nodes = [n for n in tgt_geo.nodes if n.kind == kind]

        global_translation = estimate_global_translation(ref_nodes, tgt_nodes)
        cost = build_node_cost_matrix(ref_nodes, tgt_nodes, ref_degree, tgt_degree, global_translation)
        pairs = hungarian_algorithm(cost)

        for i, j in pairs:
            mapping[ref_nodes[i].node_id] = tgt_nodes[j].node_id

    return mapping


# ==============================================================================
# Segment Mapping by Node Mapping
# ==============================================================================

def build_segment_lookup_by_node_pair(segments: List[Segment]) -> Dict[Tuple[str, str], Segment]:
    lookup: Dict[Tuple[str, str], Segment] = {}
    for seg in segments:
        key = tuple(sorted((seg.from_node, seg.to_node)))
        lookup[key] = seg
    return lookup


def compute_segment_transform(ref_seg: Segment, tgt_seg: Segment) -> SegmentMatchResult:
    delta_theta_deg = normalize_angle_diff_deg(tgt_seg.angle_deg - ref_seg.angle_deg)
    delta_theta_rad = math.radians(delta_theta_deg)

    midpoint_translation = vec_sub(tgt_seg.midpoint, ref_seg.midpoint)

    R = rotation_matrix(delta_theta_rad)
    ref_mid = point_to_np(ref_seg.midpoint)
    tgt_mid = point_to_np(tgt_seg.midpoint)
    rigid_t = tgt_mid - R @ ref_mid

    length_ratio = tgt_seg.length / max(ref_seg.length, 1e-9)

    return SegmentMatchResult(
        ref_segment=ref_seg.seg_id,
        target_segment=tgt_seg.seg_id,
        ref_from_node=ref_seg.from_node,
        ref_to_node=ref_seg.to_node,
        target_from_node=tgt_seg.from_node,
        target_to_node=tgt_seg.to_node,

        rotation_deg=round(delta_theta_deg, 6),
        translation_midpoint_xy=round_point(midpoint_translation, 6),
        rigid_translation_xy=round_point((float(rigid_t[0]), float(rigid_t[1])), 6),

        length_ref=round(ref_seg.length, 6),
        length_target=round(tgt_seg.length, 6),
        length_ratio=round(length_ratio, 6),

        midpoint_ref=round_point(ref_seg.midpoint, 6),
        midpoint_target=round_point(tgt_seg.midpoint, 6),
        angle_ref_deg=round(ref_seg.angle_deg, 6),
        angle_target_deg=round(tgt_seg.angle_deg, 6),
    )


def match_segments_via_nodes(
    ref_geo: GeoData,
    tgt_geo: GeoData,
    node_mapping: Dict[str, str],
) -> List[SegmentMatchResult]:
    tgt_lookup = build_segment_lookup_by_node_pair(tgt_geo.segments)
    results: List[SegmentMatchResult] = []

    for ref_seg in ref_geo.segments:
        tgt_from = node_mapping.get(ref_seg.from_node)
        tgt_to = node_mapping.get(ref_seg.to_node)
        if tgt_from is None or tgt_to is None:
            continue

        key = tuple(sorted((tgt_from, tgt_to)))
        tgt_seg = tgt_lookup.get(key)
        if tgt_seg is None:
            continue

        result = compute_segment_transform(ref_seg, tgt_seg)
        results.append(result)

    results.sort(key=lambda x: x.ref_segment)
    return results


# ==============================================================================
# Visualization
# ==============================================================================
def color_from_index(index: int) -> Tuple[int, int, int]:
    """
    为每一对匹配生成稳定且区分度较高的 BGR 颜色。
    共 30 种，通常足够用了。
    """
    palette = [
        (230,  99,  71), ( 60, 179, 113), ( 65, 105, 225), (255, 140,   0), (148,   0, 211),
        ( 64, 224, 208), (220,  20,  60), (255, 215,   0), (106,  90, 205), (  0, 191, 255),
        (154, 205,  50), (255, 105, 180), (205,  92,  92), ( 72, 209, 204), (123, 104, 238),
        ( 46, 139,  87), (255, 160, 122), ( 30, 144, 255), (218, 112, 214), (189, 183, 107),
        (255,  69,   0), ( 95, 158, 160), (199,  21, 133), (124, 252,   0), ( 70, 130, 180),
        (210, 105,  30), (186,  85, 211), (100, 149, 237), (233, 150, 122), (143, 188, 143),
    ]
    return palette[index % len(palette)]


def make_canvas(width: int, height: int, color: int = 255) -> np.ndarray:
    return np.full((height, width, 3), color, dtype=np.uint8)


def draw_inner_rect(image: np.ndarray, rect: Tuple[int, int, int, int], offset_x: int = 0) -> None:
    left, top, right, bottom = rect
    cv2.rectangle(
        image,
        (int(left + offset_x), int(top)),
        (int(right + offset_x), int(bottom)),
        (200, 200, 200),
        1,
        lineType=cv2.LINE_AA,
    )


def draw_nodes(
    image: np.ndarray,
    nodes: List[Node],
    offset_x: int = 0,
    show_text: bool = False,
) -> None:
    for node in nodes:
        x = int(round(node.point[0] + offset_x))
        y = int(round(node.point[1]))
        color = (0, 180, 0) if node.kind == "crossing" else (200, 180, 0)
        cv2.circle(image, (x, y), 4, color, -1, lineType=cv2.LINE_AA)
        if show_text:
            cv2.putText(
                image,
                node.node_id,
                (x + 4, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (60, 60, 60),
                1,
                lineType=cv2.LINE_AA,
            )


def draw_segments(
    image: np.ndarray,
    segments: List[Segment],
    offset_x: int = 0,
    color: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
    show_text: bool = False,
) -> None:
    for seg in segments:
        p0 = (int(round(seg.p0[0] + offset_x)), int(round(seg.p0[1])))
        p1 = (int(round(seg.p1[0] + offset_x)), int(round(seg.p1[1])))
        cv2.line(image, p0, p1, color, thickness, lineType=cv2.LINE_AA)

        if show_text:
            mx = int(round(seg.midpoint[0] + offset_x))
            my = int(round(seg.midpoint[1]))
            cv2.putText(
                image,
                seg.seg_id,
                (mx + 3, my - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (80, 80, 80),
                1,
                lineType=cv2.LINE_AA,
            )


def draw_node_matches(
    canvas: np.ndarray,
    ref_geo: GeoData,
    tgt_geo: GeoData,
    node_mapping: Dict[str, str],
    right_offset_x: int,
) -> None:
    ref_nodes = {n.node_id: n for n in ref_geo.nodes}
    tgt_nodes = {n.node_id: n for n in tgt_geo.nodes}

    for ref_id, tgt_id in node_mapping.items():
        rn = ref_nodes.get(ref_id)
        tn = tgt_nodes.get(tgt_id)
        if rn is None or tn is None:
            continue

        p0 = (int(round(rn.point[0])), int(round(rn.point[1])))
        p1 = (int(round(tn.point[0] + right_offset_x)), int(round(tn.point[1])))

        color = (220, 220, 220) if rn.kind == "crossing" else (235, 215, 160)
        cv2.line(canvas, p0, p1, color, 1, lineType=cv2.LINE_AA)


def draw_segment_match_overlay(
    canvas: np.ndarray,
    ref_geo: GeoData,
    tgt_geo: GeoData,
    matches: List[SegmentMatchResult],
    right_offset_x: int,
) -> None:
    ref_seg_map = {s.seg_id: s for s in ref_geo.segments}
    tgt_seg_map = {s.seg_id: s for s in tgt_geo.segments}

    for idx, item in enumerate(matches):
        ref_seg = ref_seg_map[item.ref_segment]
        tgt_seg = tgt_seg_map[item.target_segment]

        match_color = color_from_index(idx)

        # 左边参考段：与右边目标段同色
        rp0 = (int(round(ref_seg.p0[0])), int(round(ref_seg.p0[1])))
        rp1 = (int(round(ref_seg.p1[0])), int(round(ref_seg.p1[1])))
        cv2.line(canvas, rp0, rp1, match_color, 3, lineType=cv2.LINE_AA)

        # 右边目标段：与左边参考段同色
        tp0 = (int(round(tgt_seg.p0[0] + right_offset_x)), int(round(tgt_seg.p0[1])))
        tp1 = (int(round(tgt_seg.p1[0] + right_offset_x)), int(round(tgt_seg.p1[1])))
        cv2.line(canvas, tp0, tp1, match_color, 3, lineType=cv2.LINE_AA)

        # 两侧中点各画一个小圆点，帮助定位
        rm = (int(round(ref_seg.midpoint[0])), int(round(ref_seg.midpoint[1])))
        tm = (int(round(tgt_seg.midpoint[0] + right_offset_x)), int(round(tgt_seg.midpoint[1])))
        cv2.circle(canvas, rm, 3, match_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, tm, 3, match_color, -1, lineType=cv2.LINE_AA)

        # 标注 segment id，也用同色
        cv2.putText(
            canvas,
            item.ref_segment,
            (rm[0] + 4, rm[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            match_color,
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            item.target_segment,
            (tm[0] + 4, tm[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            match_color,
            1,
            lineType=cv2.LINE_AA,
        )


def add_title(canvas: np.ndarray, text: str, x: int, y: int = 24) -> None:
    cv2.putText(
        canvas,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (30, 30, 30),
        2,
        lineType=cv2.LINE_AA,
    )


def render_comparison_image(
    ref_geo: GeoData,
    tgt_geo: GeoData,
    node_mapping: Dict[str, str],
    segment_matches: List[SegmentMatchResult],
    out_path: Path,
) -> None:
    panel_gap = 120
    legend_h = 70
    width = ref_geo.width + tgt_geo.width + panel_gap
    height = max(ref_geo.height, tgt_geo.height) + legend_h

    canvas = make_canvas(width, height, color=255)

    left_offset_x = 0
    right_offset_x = ref_geo.width + panel_gap

    # titles
    add_title(canvas, "Reference", 20, 28)
    add_title(canvas, "Target", right_offset_x + 20, 28)

    # shift drawings below title
    body = canvas[legend_h:, :, :]
    draw_inner_rect(body, ref_geo.inner_rect, offset_x=left_offset_x)
    draw_inner_rect(body, tgt_geo.inner_rect, offset_x=right_offset_x)

    draw_segments(body, ref_geo.segments, offset_x=left_offset_x, color=(200, 200, 200), thickness=1, show_text=False)
    draw_segments(body, tgt_geo.segments, offset_x=right_offset_x, color=(200, 200, 200), thickness=1, show_text=False)

    draw_node_matches(body, ref_geo, tgt_geo, node_mapping, right_offset_x)
    draw_nodes(body, ref_geo.nodes, offset_x=left_offset_x, show_text=False)
    draw_nodes(body, tgt_geo.nodes, offset_x=right_offset_x, show_text=False)
    draw_segment_match_overlay(body, ref_geo, tgt_geo, segment_matches, right_offset_x)

    # legend
    cv2.putText(canvas, "Gray lines: all segments", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Same color = one matched segment pair", (240, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, "30 colors reused cyclically", (560, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), canvas)
    if not ok:
        raise RuntimeError(f"Cannot write image: {out_path}")


# ==============================================================================
# Main compare
# ==============================================================================

def compare_one_target(ref_path: Path, target_path: Path) -> Tuple[dict, Dict[str, str], List[SegmentMatchResult], GeoData, GeoData]:
    ref_geo = load_geo_json(ref_path)
    tgt_geo = load_geo_json(target_path)

    node_mapping = match_nodes_by_kind(ref_geo, tgt_geo)
    segment_matches = match_segments_via_nodes(ref_geo, tgt_geo, node_mapping)

    payload = {
        "reference_file": str(ref_path),
        "target_file": str(target_path),
        "reference_node_count": len(ref_geo.nodes),
        "target_node_count": len(tgt_geo.nodes),
        "reference_segment_count": len(ref_geo.segments),
        "target_segment_count": len(tgt_geo.segments),
        "matched_node_count": len(node_mapping),
        "matched_segment_count": len(segment_matches),
        "node_mapping": node_mapping,
        "segment_matches": [asdict(item) for item in segment_matches],
    }
    return payload, node_mapping, segment_matches, ref_geo, tgt_geo


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match nodes first, then infer segment correspondence and visualize transforms."
    )
    parser.add_argument("--ref", required=True, help="Reference json, e.g. output/centerline/geo_1.json")
    parser.add_argument(
        "--targets",
        nargs="+",
        required=True,
        help="Target json files, e.g. output/centerline/geo_2.json output/centerline/geo_3.json",
    )
    parser.add_argument("--output-dir", default="output/compare_geo", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ref_path = Path(args.ref)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for target in args.targets:
        target_path = Path(target)

        payload, node_mapping, segment_matches, ref_geo, tgt_geo = compare_one_target(ref_path, target_path)

        json_name = f"{ref_path.stem}_vs_{target_path.stem}.json"
        img_name = f"{ref_path.stem}_vs_{target_path.stem}_vis.png"

        json_path = output_dir / json_name
        img_path = output_dir / img_name

        save_json(json_path, payload)
        render_comparison_image(ref_geo, tgt_geo, node_mapping, segment_matches, img_path)

        print(f"[OK] {target_path.name}")
        print(f"  matched_node_count    = {payload['matched_node_count']}")
        print(f"  matched_segment_count = {payload['matched_segment_count']}")
        print(f"  json                  = {json_path}")
        print(f"  image                 = {img_path}")

        for item in payload["segment_matches"]:
            print(
                f"  {item['ref_segment']:>4s} -> {item['target_segment']:<4s} | "
                f"rot={item['rotation_deg']:>9.4f} deg | "
                f"mid_t=({item['translation_midpoint_xy'][0]:>8.2f}, {item['translation_midpoint_xy'][1]:>8.2f}) | "
                f"len_ratio={item['length_ratio']:.4f}"
            )
        print()


if __name__ == "__main__":
    main()
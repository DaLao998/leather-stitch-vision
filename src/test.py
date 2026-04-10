import cv2
import numpy as np
import time

# =========================================
# 1. ROI：定义有效区域多边形
# =========================================
def get_crop_local_polygon():
    return np.array([
        [105,   0],
        [1630,  27],
        [1596, 2244],
        [0,    2181]
    ], dtype=np.int32)

def is_line_in_roi(x1, y1, x2, y2, poly):
    """判断线段是否在 ROI 多边形内（只要有一个端点在内或线上即可）"""
    pt1_test = cv2.pointPolygonTest(poly, (float(x1), float(y1)), False)
    pt2_test = cv2.pointPolygonTest(poly, (float(x2), float(y2)), False)
    return pt1_test >= 0 or pt2_test >= 0


# =========================================
# 2. 改进的直线合并算法（法向距离判定）
# =========================================
def merge_similar_lines(lines, angle_thresh_deg=5, perp_dist_thresh=15):
    """
    通过角度和端点到直线的垂直距离，将共线的碎线段合并为长线
    """
    if lines is None or len(lines) == 0:
        return []

    line_info = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        dx = x2 - x1
        dy = y2 - y1
        
        # 计算角度 (0 到 180 度)
        angle = np.degrees(np.arctan2(dy, dx))
        if angle < 0:
            angle += 180
            
        # 计算直线一般式 Ax + By + C = 0
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        norm = np.hypot(A, B)
        
        if norm == 0: # 忽略退化为点的线段
            continue 

        line_info.append({
            "pts": (x1, y1, x2, y2),
            "angle": angle,
            "eq": (A/norm, B/norm, C/norm) # 归一化方程参数，便于算距离
        })

    used = [False] * len(line_info)
    merged = []

    for i in range(len(line_info)):
        if used[i]:
            continue

        group = [line_info[i]]
        used[i] = True
        A_i, B_i, C_i = line_info[i]["eq"]

        for j in range(i + 1, len(line_info)):
            if used[j]:
                continue

            a1 = line_info[i]["angle"]
            a2 = line_info[j]["angle"]
            
            # 处理 0度与180度 的边缘跳跃
            da = abs(a1 - a2)
            da = min(da, 180 - da)

            # 计算线段 j 的两个端点到主直线 i 的垂直距离
            xj1, yj1, xj2, yj2 = line_info[j]["pts"]
            dist1 = abs(A_i * xj1 + B_i * yj1 + C_i)
            dist2 = abs(A_i * xj2 + B_i * yj2 + C_i)

            # 如果角度接近，且偏移距离极小，认为是同一条直线
            if da < angle_thresh_deg and dist1 < perp_dist_thresh and dist2 < perp_dist_thresh:
                group.append(line_info[j])
                used[j] = True

        # ==========================================
        # 生成合并后的新直线：寻找组内距离最远的两个端点
        # ==========================================
        pts = []
        for g in group:
            x1, y1, x2, y2 = g["pts"]
            pts.append((x1, y1))
            pts.append((x2, y2))
            
        if len(pts) == 2:
            merged.append((int(pts[0][0]), int(pts[0][1]), int(pts[1][0]), int(pts[1][1])))
            continue

        max_dist = 0
        best_pair = (pts[0], pts[1])
        # 穷举两两端点，找到跨度最大的对作为新线段两头
        for p1_idx in range(len(pts)):
            for p2_idx in range(p1_idx + 1, len(pts)):
                d = np.hypot(pts[p1_idx][0] - pts[p2_idx][0], pts[p1_idx][1] - pts[p2_idx][1])
                if d > max_dist:
                    max_dist = d
                    best_pair = (pts[p1_idx], pts[p2_idx])

        p1, p2 = best_pair
        merged.append((int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))

    return merged


# =========================================
# 3. 提取主函数
# =========================================
def extract_pattern_lines_optimized(
    image_path: str,
    out_line_path: str = "pattern_lines_optimized.png",
    bin_thresh: int = 127,
    min_line_length: int = 100,
    max_line_gap: int = 50,
    hough_threshold: int = 60
):
    # 1. 读图
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    # 2. 二值化：反相，使得原图黑色的通道变成白色的粗线条 (255)
    _, binary = cv2.threshold(img, bin_thresh, 255, cv2.THRESH_BINARY_INV)

    # 3. 提取骨架 (Zhang-Suen 算法，极其关键，防止毛刺)
    # 此步骤对二值图全局进行，不切掉边缘
    skeleton = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    # 4. 霍夫直线检测 (在全局骨架上检测)
    raw_lines = cv2.HoughLinesP(
        skeleton,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    # 5. 合并共线碎线段
    merged_lines = merge_similar_lines(
        raw_lines, 
        angle_thresh_deg=5, 
        perp_dist_thresh=15  # 允许与主线有15个像素内的偏移容错
    )

    # 6. ROI 过滤与可视化
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    poly = get_crop_local_polygon()
    
    # 绘制半透明的 ROI 区域边界用于确认
    cv2.polylines(vis, [poly], isClosed=True, color=(255, 200, 0), thickness=2)

    final_valid_lines = []
    for x1, y1, x2, y2 in merged_lines:
        # 只保留落在实际工作区内的花样线
        if is_line_in_roi(x1, y1, x2, y2, poly):
            final_valid_lines.append((x1, y1, x2, y2))
            # 绘制合并后的长线（加粗显示）
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.imwrite(out_line_path, vis)

    return {
        "final_lines": final_valid_lines,
        "skeleton": skeleton,
        "result_image": vis
    }

# =========================================
# 运行测试
# =========================================
if __name__ == "__main__":
    total_start = time.perf_counter()
    image_path = "./output/pattern/1_matrix_instances_bw.png" # 替换为你的路径
    
    try:
        result = extract_pattern_lines_optimized(
            image_path=image_path,
            out_line_path="pattern_lines_optimized.png"
        )
        print(f"检测并合并后，得到有效的贯穿花样线数: {len(result['final_lines'])}")
        for i, line in enumerate(result["final_lines"], 1):
            print(f"Pattern Line {i}: {line}")
        total_time = time.perf_counter() - total_start
        print(f"{total_time}")
            
    except Exception as e:
        print(f"执行出错: {e}")
import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from scipy.optimize import least_squares

# 与 calibrate.py 常见后缀一致，用于扫描目录
IMAGE_EXTS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".dib", ".tif", ".tiff", ".webp",
)


def list_images_in_dir(directory: str) -> List[str]:
    """目录下（不含子目录）按文件名排序的图像路径列表。"""
    directory = os.path.normpath(directory)
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"不是目录：{directory}")
    exts = set(IMAGE_EXTS)
    files: List[str] = []
    for name in sorted(os.listdir(directory)):
        ext = os.path.splitext(name)[1].lower()
        if ext in exts:
            files.append(os.path.join(directory, name))
    return files


def append_jsonl_line(jsonl_path: str, record: Dict) -> None:
    """追加一行 JSON（UTF-8），即 jsonl；将 numpy 标量转为 Python 类型以便序列化。"""

    def _sanitize(obj: object) -> object:
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, tuple):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    safe = _sanitize(record)
    line = json.dumps(safe, ensure_ascii=False) + "\n"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(line)


@dataclass
class LaserSpotConfig:
    # red：适合红色全站仪激光点；bright：适合颜色不稳定但亮度突出的光斑
    mode: str = "red"

    # 连通域面积范围，需要根据相机分辨率和光斑大小调整
    min_area: int = 10
    max_area: int = 50000

    # 阈值分割参数
    percentile: float = 99.3
    min_threshold: int = 10

    # 形态学参数
    open_kernel: int = 3
    close_kernel: int = 5

    # Gaussian 拟合参数
    use_gaussian: bool = True
    gaussian_roi_half_size: int = 25
    gaussian_core_ratio: float = 0.15

    # 如果饱和像素太多，Gaussian 拟合会不可靠
    saturation_value: int = 250
    max_saturation_ratio_for_gaussian: float = 0.35

    # 调试输出
    save_debug: bool = True

#限定roi
def _clip_xywh(
    x: int, y: int, w: int, h: int, img_w: int, img_h: int
) -> Optional[Tuple[int, int, int, int]]:
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(0, min(w, img_w - x))
    h = max(0, min(h, img_h - y))
    if w < 2 or h < 2:
        return None
    return x, y, w, h


def _ascii_overlay_text(text: str, max_len: int = 56) -> str:
    """仅保留可打印 ASCII，便于 cv2.putText；其余替换为下划线。"""
    out: List[str] = []
    for ch in text:
        if ch.isascii() and 32 <= ord(ch) < 127:
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    return s[:max_len] if len(s) > max_len else s


def select_rois_interactive(
    img_bgr: np.ndarray,
    index: int = 1,
    total: int = 1,
    basename: str = "",
    full_path: str = "",
) -> Optional[List[Tuple[int, int, int, int]]]:
    """
    原图 1:1 窗口，可多框 ROI。顶部有信息条（ASCII 文件名 + 张数序号）。

    full_path：在终端打印完整路径（UTF-8），便于核对中文路径。

    返回 None：ESC 放弃（本张不限制 ROI）；返回 []：Enter 且无矩形（同上）；
    非空 list：各矩形 (x,y,w,h) 在原图坐标系中。
    """
    BAR_H = 44
    win = f"ROI-{index}-{total}"
    img_h, img_w = img_bgr.shape[:2]

    state: Dict = {
        "rois": [],
        "dragging": False,
        "x0": 0,
        "y0": 0,
        "x1": 0,
        "y1": 0,
    }

    def draw_overlay(vis: np.ndarray) -> None:
        cv2.rectangle(vis, (0, 0), (img_w - 1, BAR_H - 1), (0, 0, 0), -1)
        line1 = f"{index} / {total}"
        line2 = _ascii_overlay_text(basename) if basename else line1
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis, line1, (6, 18), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, line2, (6, 36), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    def redraw() -> None:
        vis = img_bgr.copy()
        draw_overlay(vis)
        for rx, ry, rw, rh in state["rois"]:
            cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 1)
        if state["dragging"]:
            ax0, ay0 = state["x0"], state["y0"]
            ax1, ay1 = state["x1"], state["y1"]
            tlx, tly = min(ax0, ax1), min(ay0, ay1)
            brx, bry = max(ax0, ax1), max(ay0, ay1)
            cv2.rectangle(vis, (tlx, tly), (brx, bry), (0, 255, 255), 1)
        cv2.imshow(win, vis)

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        x = int(np.clip(x, 0, img_w - 1))
        y = int(np.clip(y, 0, img_h - 1))
        if event == cv2.EVENT_LBUTTONDOWN:
            state["dragging"] = True
            state["x0"] = state["x1"] = x
            state["y0"] = state["y1"] = y
            redraw()
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["x1"], state["y1"] = x, y
            redraw()
        elif event == cv2.EVENT_LBUTTONUP and state["dragging"]:
            state["dragging"] = False
            state["x1"], state["y1"] = x, y
            tlx = min(state["x0"], state["x1"])
            tly = min(state["y0"], state["y1"])
            brx = max(state["x0"], state["x1"])
            bry = max(state["y0"], state["y1"])
            tly = max(tly, BAR_H)
            rw = brx - tlx
            rh = bry - tly
            clipped = _clip_xywh(tlx, tly, rw, rh, img_w, img_h)
            if clipped is not None:
                state["rois"].append(clipped)
            redraw()

    if full_path:
        print(f"[ROI] 第 {index}/{total} 张 | {full_path}")
    else:
        print(f"[ROI] 第 {index}/{total} 张 | {basename}")
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    try:
        redraw()
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                return None
            if key in (13, 10):
                return list(state["rois"])
            if key in (8, 127):
                if state["rois"]:
                    state["rois"].pop()
                    redraw()
    finally:
        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass
        cv2.waitKey(1)


def imread_unicode(path: str) -> np.ndarray:
    """
    支持中文路径的图像读取。
    """
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{path}")
    return img


def imwrite_unicode(path: str, img: np.ndarray) -> None:
    """
    支持中文路径的图像保存。
    """
    ext = os.path.splitext(path)[1]
    if ext == "":
        ext = ".png"
        path += ext
    ok, data = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"无法编码图像：{path}")
    data.tofile(path)


def normalize_to_uint8(src: np.ndarray) -> np.ndarray:
    src = src.astype(np.float32)
    min_v, max_v = float(np.min(src)), float(np.max(src))
    if max_v - min_v < 1e-6:
        return np.zeros_like(src, dtype=np.uint8)
    dst = (src - min_v) / (max_v - min_v) * 255.0
    return np.clip(dst, 0, 255).astype(np.uint8)


def build_laser_score_and_mask(
    img_bgr: np.ndarray,
    cfg: LaserSpotConfig
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    构建激光点得分图 score 和二值掩码 mask。

    score:
        用于灰度加权质心和 Gaussian 拟合的强度图。
    mask:
        用于最大连通域提取的二值图。
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(img_bgr)

    if cfg.mode == "red":
        # 红色增强：R 明显大于 G/B 的区域更可能是红色激光
        red_dominance = r.astype(np.int16) - np.maximum(g, b).astype(np.int16)
        red_dominance = np.clip(red_dominance, 0, 255).astype(np.uint8)

        # HSV 红色范围：红色位于 H 的两端
        red_hsv_1 = cv2.inRange(hsv, np.array([0, 30, 20]), np.array([20, 255, 255]))
        red_hsv_2 = cv2.inRange(hsv, np.array([160, 30, 20]), np.array([180, 255, 255]))
        red_hsv = cv2.bitwise_or(red_hsv_1, red_hsv_2)

        # 分数图：优先使用红色增强，同时保留 HSV 红色区域的亮度
        score = np.maximum(red_dominance, (v * (red_hsv > 0)).astype(np.uint8))

        # Otsu + 百分位阈值结合，避免固定阈值过死
        blur = cv2.GaussianBlur(score, (3, 3), 0)
        otsu_thr, _ = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        pct_thr = np.percentile(blur, cfg.percentile)
        thr = max(float(otsu_thr), float(pct_thr) * 0.6, cfg.min_threshold)

        red_mask = (blur >= thr).astype(np.uint8) * 255

        # 处理过曝：激光中心可能从红色变成白色，因此只接纳“靠近红色区域”的高亮白点
        bright_thr = max(np.percentile(v, 99.7), 220)
        bright_mask = (v >= bright_thr).astype(np.uint8) * 255

        red_near = cv2.dilate(red_mask, np.ones((9, 9), np.uint8), iterations=1)
        white_core_mask = cv2.bitwise_and(bright_mask, red_near)

        mask = cv2.bitwise_or(red_mask, white_core_mask)

        # 对后续灰度加权与 Gaussian 拟合，红色光斑建议用 R 通道和亮度 V 的融合
        intensity = np.maximum(r, v).astype(np.uint8)

    elif cfg.mode == "bright":
        # 对颜色不稳定的激光点，直接用亮度 V
        intensity = v.copy()
        blur = cv2.GaussianBlur(intensity, (3, 3), 0)

        otsu_thr, _ = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        pct_thr = np.percentile(blur, cfg.percentile)
        thr = max(float(otsu_thr), float(pct_thr) * 0.7, cfg.min_threshold)

        mask = (blur >= thr).astype(np.uint8) * 255

    else:
        raise ValueError("cfg.mode 只能是 'red' 或 'bright'")

    # 形态学去噪：开运算去小噪声，闭运算填孔
    if cfg.open_kernel > 1:
        k_open = np.ones((cfg.open_kernel, cfg.open_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    if cfg.close_kernel > 1:
        k_close = np.ones((cfg.close_kernel, cfg.close_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    debug = {
        "mode": cfg.mode,
        "threshold_percentile": cfg.percentile,
        "mask_nonzero_pixels": int(np.count_nonzero(mask)),
    }

    return intensity, mask, debug


def select_best_connected_component(
    intensity: np.ndarray,
    mask: np.ndarray,
    cfg: LaserSpotConfig
) -> Tuple[np.ndarray, Dict]:
    """
    从二值 mask 中选择最可信的连通域。
    默认不是简单选面积最大，而是综合面积、亮度总和、峰值亮度评分。
    """
    mask_u8 = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    best_label = None
    best_score = -1.0
    components = []

    for label in range(1, num_labels):  # 0 是背景
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < cfg.min_area or area > cfg.max_area:
            continue

        comp_mask = labels == label
        values = intensity[comp_mask].astype(np.float32)

        sum_i = float(np.sum(values))
        mean_i = float(np.mean(values))
        max_i = float(np.max(values))

        # 评分：亮度总能量为主，峰值和面积为辅
        score = sum_i + 0.2 * area * max_i + 0.1 * area * mean_i

        item = {
            "label": int(label),
            "bbox": [x, y, w, h],
            "area": area,
            "opencv_centroid": [
                float(centroids[label][0]),
                float(centroids[label][1])
            ],
            "sum_intensity": sum_i,
            "mean_intensity": mean_i,
            "max_intensity": max_i,
            "score": score,
        }
        components.append(item)

        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        raise RuntimeError(
            "没有找到符合条件的激光点连通域。"
            "请调低 min_area/min_threshold，或检查图像曝光、激光颜色和背景反光。"
        )

    best_mask = (labels == best_label).astype(np.uint8) * 255
    best_info = next(c for c in components if c["label"] == int(best_label))
    best_info["num_components_after_filter"] = len(components)

    return best_mask, best_info


def gray_weighted_centroid(
    intensity: np.ndarray,
    component_mask: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[float, float, Dict]:
    """
    灰度加权质心：

        x_c = sum(x * I) / sum(I)
        y_c = sum(y * I) / sum(I)

    这里会减去局部背景，降低背景亮度对质心的偏移影响。
    """
    h_img, w_img = intensity.shape[:2]

    if bbox is None:
        ys, xs = np.where(component_mask > 0)
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    else:
        x, y, w, h = bbox
        pad = max(5, int(max(w, h) * 0.5))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w_img, x + w + pad)
        y1 = min(h_img, y + h + pad)

    roi_i = intensity[y0:y1, x0:x1].astype(np.float64)
    roi_m = component_mask[y0:y1, x0:x1] > 0

    if np.count_nonzero(roi_m) == 0:
        raise RuntimeError("连通域 mask 为空，无法计算灰度加权质心。")

    # 局部背景：优先用 mask 外区域估计；如果没有，则用 ROI 低分位数
    bg_pixels = roi_i[~roi_m]
    if bg_pixels.size >= 20:
        background = float(np.percentile(bg_pixels, 50))
    else:
        background = float(np.percentile(roi_i, 10))

    weights = np.maximum(roi_i - background, 0.0) * roi_m

    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-9:
        # 如果减背景后全为 0，则退回到不减背景的加权
        weights = roi_i * roi_m
        weight_sum = float(np.sum(weights))

    if weight_sum <= 1e-9:
        # 再失败就退回到二值几何质心
        ys, xs = np.where(roi_m)
        cx = float(x0 + np.mean(xs))
        cy = float(y0 + np.mean(ys))
        return cx, cy, {
            "method": "binary_centroid_fallback",
            "background": background,
            "weight_sum": weight_sum,
        }

    yy, xx = np.indices(weights.shape)

    cx = float(x0 + np.sum(xx * weights) / weight_sum)
    cy = float(y0 + np.sum(yy * weights) / weight_sum)

    return cx, cy, {
        "method": "gray_weighted_centroid",
        "background": background,
        "weight_sum": weight_sum,
        "roi": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
    }


def gaussian_2d_model(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    二维非旋转椭圆 Gaussian：

        I(x,y) = B + A * exp(-0.5 * [((x-x0)/sx)^2 + ((y-y0)/sy)^2])

    params = [B, A, x0, y0, sx, sy]
    """
    B, A, x0, y0, sx, sy = params
    sx = max(float(sx), 1e-6)
    sy = max(float(sy), 1e-6)

    return B + A * np.exp(
        -0.5 * (((x - x0) / sx) ** 2 + ((y - y0) / sy) ** 2)
    )


def gaussian_refine_center(
    intensity: np.ndarray,
    component_mask: np.ndarray,
    initial_center: Tuple[float, float],
    cfg: LaserSpotConfig
) -> Tuple[float, float, Dict]:
    """
    在灰度加权质心附近做二维 Gaussian 拟合。
    注意：如果光斑严重过曝或形状严重非 Gaussian，该结果可能不如灰度加权质心稳定。
    """
    h_img, w_img = intensity.shape[:2]
    cx0, cy0 = initial_center
    half = cfg.gaussian_roi_half_size

    x0 = max(0, int(round(cx0)) - half)
    y0 = max(0, int(round(cy0)) - half)
    x1 = min(w_img, int(round(cx0)) + half + 1)
    y1 = min(h_img, int(round(cy0)) + half + 1)

    roi_i = intensity[y0:y1, x0:x1].astype(np.float64)
    roi_m = component_mask[y0:y1, x0:x1] > 0

    if roi_i.size < 25 or np.count_nonzero(roi_m) < 8:
        return cx0, cy0, {
            "method": "gaussian_skipped",
            "reason": "ROI 或有效像素太少",
            "success": False,
        }

    # 背景估计
    bg_pixels = roi_i[~roi_m]
    if bg_pixels.size >= 20:
        background = float(np.percentile(bg_pixels, 50))
    else:
        background = float(np.percentile(roi_i, 10))

    signal = np.maximum(roi_i - background, 0.0)
    peak = float(np.max(signal))

    if peak <= 1e-6:
        return cx0, cy0, {
            "method": "gaussian_skipped",
            "reason": "峰值信号太弱",
            "success": False,
            "background": background,
        }

    # 饱和判断：过曝太多时 Gaussian 拟合容易偏
    valid_for_sat = roi_m
    saturation_ratio = float(
        np.mean(roi_i[valid_for_sat] >= cfg.saturation_value)
    ) if np.count_nonzero(valid_for_sat) > 0 else 0.0

    if saturation_ratio > cfg.max_saturation_ratio_for_gaussian:
        return cx0, cy0, {
            "method": "gaussian_skipped",
            "reason": "饱和像素比例过高，Gaussian 拟合不可靠",
            "success": False,
            "background": background,
            "saturation_ratio": saturation_ratio,
        }

    # 为了降低不规则拖尾影响，只用核心区域参与 Gaussian 拟合
    core_threshold = max(cfg.gaussian_core_ratio * peak, 1.0)
    fit_mask = roi_m & (signal >= core_threshold)

    # 如果核心区域太少，则退回使用整个连通域
    if np.count_nonzero(fit_mask) < 12:
        fit_mask = roi_m & (signal > 0)

    if np.count_nonzero(fit_mask) < 12:
        return cx0, cy0, {
            "method": "gaussian_skipped",
            "reason": "可用于拟合的像素太少",
            "success": False,
            "background": background,
        }

    yy, xx = np.indices(roi_i.shape)
    xs = xx[fit_mask].astype(np.float64)
    ys = yy[fit_mask].astype(np.float64)
    zs = roi_i[fit_mask].astype(np.float64)

    # 初值
    local_cx0 = cx0 - x0
    local_cy0 = cy0 - y0
    A0 = max(float(np.max(zs) - background), 1.0)
    B0 = max(background, 0.0)

    # 用二阶矩估计 sigma 初值
    w = np.maximum(zs - background, 0.0)
    if np.sum(w) > 1e-9:
        sx0 = np.sqrt(np.sum(((xs - local_cx0) ** 2) * w) / np.sum(w))
        sy0 = np.sqrt(np.sum(((ys - local_cy0) ** 2) * w) / np.sum(w))
    else:
        sx0 = sy0 = 3.0

    sx0 = float(np.clip(sx0, 1.0, max(roi_i.shape)))
    sy0 = float(np.clip(sy0, 1.0, max(roi_i.shape)))

    p0 = np.array([B0, A0, local_cx0, local_cy0, sx0, sy0], dtype=np.float64)

    lower = np.array([0.0, 1.0, 0.0, 0.0, 0.5, 0.5], dtype=np.float64)
    upper = np.array([
        255.0,
        255.0,
        roi_i.shape[1] - 1.0,
        roi_i.shape[0] - 1.0,
        max(roi_i.shape),
        max(roi_i.shape)
    ], dtype=np.float64)

    def residual(params: np.ndarray) -> np.ndarray:
        pred = gaussian_2d_model(params, xs, ys)
        return pred - zs

    try:
        res = least_squares(
            residual,
            p0,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=3.0,
            max_nfev=1000,
        )
    except Exception as e:
        return cx0, cy0, {
            "method": "gaussian_failed",
            "reason": str(e),
            "success": False,
            "background": background,
        }

    B, A, fit_x, fit_y, sx, sy = res.x

    gx = float(x0 + fit_x)
    gy = float(y0 + fit_y)

    # 合理性检查：拟合中心不能离初始中心太远
    shift = float(np.hypot(gx - cx0, gy - cy0))
    max_allowed_shift = max(3.0, cfg.gaussian_roi_half_size * 0.5)

    if (not res.success) or shift > max_allowed_shift:
        return cx0, cy0, {
            "method": "gaussian_rejected",
            "reason": "拟合未收敛或中心漂移过大",
            "success": bool(res.success),
            "shift_from_weighted": shift,
            "cost": float(res.cost),
            "background": background,
        }

    return gx, gy, {
        "method": "gaussian_2d_refine",
        "success": bool(res.success),
        "background": background,
        "amplitude": float(A),
        "sigma_x": float(sx),
        "sigma_y": float(sy),
        "cost": float(res.cost),
        "nfev": int(res.nfev),
        "fit_roi": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
        "fit_pixels": int(np.count_nonzero(fit_mask)),
        "saturation_ratio": saturation_ratio,
        "shift_from_weighted": shift,
    }


def annotate_result(
    img_bgr: np.ndarray,
    component_mask: np.ndarray,
    weighted_center: Tuple[float, float],
    final_center: Tuple[float, float],
    bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    vis = img_bgr.copy()

    x, y, w, h = bbox
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

    contours, _ = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(vis, contours, -1, (255, 0, 0), 1)

    wx, wy = weighted_center
    gx, gy = final_center

    cv2.drawMarker(
        vis,
        (int(round(wx)), int(round(wy))),
        (0, 255, 0),
        cv2.MARKER_CROSS,
        25,
        2,
    )
    cv2.putText(
        vis,
        f"Weighted ({wx:.2f}, {wy:.2f})",
        (int(round(wx)) + 10, int(round(wy)) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    cv2.drawMarker(
        vis,
        (int(round(gx)), int(round(gy))),
        (0, 0, 255),
        cv2.MARKER_TILTED_CROSS,
        30,
        2,
    )
    cv2.putText(
        vis,
        f"Final ({gx:.2f}, {gy:.2f})",
        (int(round(gx)) + 10, int(round(gy)) + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
    )

    return vis


def annotate_final_cross_only(
    img_bgr: np.ndarray,
    center_xy: Tuple[float, float],
    half_arm: int = 4,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 1,
) -> np.ndarray:
    """
    在原图上只画最终中心：极短十字丝（亚像素中心四舍五入到整像素后画线）。
    half_arm：单侧臂长（像素），总长约 2*half_arm+1。
    """
    vis = img_bgr.copy()
    ix = int(round(center_xy[0]))
    iy = int(round(center_xy[1]))
    h_img, w_img = vis.shape[:2]
    ha = max(1, int(half_arm))

    x0 = max(0, ix - ha)
    x1 = min(w_img - 1, ix + ha)
    y0 = max(0, iy - ha)
    y1 = min(h_img - 1, iy + ha)

    cv2.line(vis, (x0, iy), (x1, iy), color, thickness, cv2.LINE_AA)
    cv2.line(vis, (ix, y0), (ix, y1), color, thickness, cv2.LINE_AA)
    return vis


def locate_laser_spot(
    image_path: str,
    output_dir: Optional[str] = None,
    cfg: Optional[LaserSpotConfig] = None,
    select_roi: bool = False,
    roi_xywh_list: Optional[List[Tuple[int, int, int, int]]] = None,
    write_per_image_json: bool = True,
    batch_index: Optional[int] = None,
    batch_total: Optional[int] = None,
) -> Dict:
    """
    主流程：
    1. 读取图像
    2. 可选：交互框选多个 ROI（整图 1:1；分割 mask 与 ROI 并集求交），或由参数 roi_xywh_list 指定
    3. 颜色/亮度分割
    4. 最大/最可信连通域提取
    5. 灰度加权质心
    6. Gaussian 拟合细化
    7. 保存调试图 / 可选每张 JSON
    """
    if cfg is None:
        cfg = LaserSpotConfig()

    img_full = imread_unicode(image_path)
    roi_xywh_list_effective: Optional[List[Tuple[int, int, int, int]]] = None

    if roi_xywh_list is not None:
        roi_xywh_list_effective = list(roi_xywh_list)
    elif select_roi:
        idx = int(batch_index) if batch_index is not None else 1
        tot = int(batch_total) if batch_total is not None else 1
        picked = select_rois_interactive(
            img_full,
            index=idx,
            total=tot,
            basename=os.path.basename(image_path),
            full_path=os.path.abspath(image_path),
        )
        if picked is not None and len(picked) > 0:
            roi_xywh_list_effective = picked

    img = img_full

    intensity, mask, mask_debug = build_laser_score_and_mask(img, cfg)

    if roi_xywh_list_effective:
        h, w = img_full.shape[:2]
        roi_union = np.zeros((h, w), dtype=np.uint8)
        for rx, ry, rw, rh in roi_xywh_list_effective:
            clipped = _clip_xywh(rx, ry, rw, rh, w, h)
            if clipped is None:
                continue
            rx, ry, rw, rh = clipped
            roi_union[ry : ry + rh, rx : rx + rw] = 255
        mask = cv2.bitwise_and(mask, roi_union)
        mask_debug = dict(mask_debug)
        mask_debug["roi_rect_count"] = len(roi_xywh_list_effective)

    component_mask, comp_info = select_best_connected_component(
        intensity, mask, cfg
    )

    x, y, w, h = comp_info["bbox"]
    weighted_x, weighted_y, weighted_info = gray_weighted_centroid(
        intensity,
        component_mask,
        bbox=(x, y, w, h)
    )

    final_x, final_y = weighted_x, weighted_y
    gaussian_info = {
        "method": "gaussian_not_used",
        "success": False,
    }

    if cfg.use_gaussian:
        final_x, final_y, gaussian_info = gaussian_refine_center(
            intensity,
            component_mask,
            initial_center=(weighted_x, weighted_y),
            cfg=cfg
        )

    result = {
        "image_path": image_path,
        "image_basename": os.path.basename(image_path),
        "final_center_xy": [float(final_x), float(final_y)],
        "weighted_center_xy": [float(weighted_x), float(weighted_y)],
        "component": comp_info,
        "weighted_centroid_info": weighted_info,
        "gaussian_info": gaussian_info,
        "mask_debug": mask_debug,
        "config": asdict(cfg),
        "roi_xywh_list": (
            [[int(a), int(b), int(c), int(d)] for a, b, c, d in roi_xywh_list_effective]
            if roi_xywh_list_effective
            else None
        ),
        "roi_xywh": (
            [
                int(roi_xywh_list_effective[0][0]),
                int(roi_xywh_list_effective[0][1]),
                int(roi_xywh_list_effective[0][2]),
                int(roi_xywh_list_effective[0][3]),
            ]
            if roi_xywh_list_effective and len(roi_xywh_list_effective) == 1
            else None
        ),
        "batch_index": batch_index,
        "batch_total": batch_total,
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(image_path))[0]

        if cfg.save_debug:
            imwrite_unicode(
                os.path.join(output_dir, f"{base}_01_intensity.png"),
                normalize_to_uint8(intensity)
            )
            imwrite_unicode(
                os.path.join(output_dir, f"{base}_02_mask.png"),
                mask
            )
            imwrite_unicode(
                os.path.join(output_dir, f"{base}_03_component.png"),
                component_mask
            )

            vis_img = img_full
            component_mask_vis = component_mask
            bbox_vis = (x, y, w, h)

            vis = annotate_result(
                vis_img,
                component_mask_vis,
                weighted_center=(weighted_x, weighted_y),
                final_center=(final_x, final_y),
                bbox=bbox_vis,
            )
            if roi_xywh_list_effective:
                for rx, ry, rw, rh in roi_xywh_list_effective:
                    cv2.rectangle(
                        vis, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 2
                    )
            imwrite_unicode(
                os.path.join(output_dir, f"{base}_04_result.png"),
                vis
            )

        final_only_path = os.path.join(
            output_dir, f"{base}_05_final_cross.png"
        )
        final_only = annotate_final_cross_only(
            img_full,
            (final_x, final_y),
            half_arm=4,
        )
        imwrite_unicode(final_only_path, final_only)
        result["output_final_cross_png"] = final_only_path

        if write_per_image_json:
            json_path = os.path.join(output_dir, f"{base}_center_result.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            result["output_json"] = json_path
        else:
            result["output_json"] = None

    return result


def main():
    parser = argparse.ArgumentParser(
        description="全站仪激光点中心定位：颜色/亮度分割 + 最大连通域 + 灰度加权质心 + Gaussian 拟合"
    )
    parser.add_argument("--image", default=None, help="单张图像路径")
    parser.add_argument(
        "--input-dir",
        default=None,
        dest="input_dir",
        help="处理文件夹内全部图像（按文件名排序）；每张先框 ROI 再算下一张；汇总写入 center_result.jsonl",
    )
    parser.add_argument("--out", default="laser_debug", help="输出调试目录")
    parser.add_argument("--mode", default="red", choices=["red", "bright"],
                        help="red 适合红色激光；bright 适合亮点检测")
    parser.add_argument("--min-area", type=int, default=10)
    parser.add_argument("--max-area", type=int, default=50000)
    parser.add_argument("--no-gaussian", action="store_true",
                        help="只输出灰度加权质心，不做 Gaussian 拟合")
    parser.add_argument(
        "--select-roi",
        action="store_true",
        help="单张模式 (--image)：处理前 1:1 窗口多框 ROI；Enter 结束，ESC 放弃，Backspace 撤销上一块",
    )
    args = parser.parse_args()

    if not args.image and not args.input_dir:
        parser.error("请指定 --image 或 --input-dir 之一")
    if args.image and args.input_dir:
        parser.error("--image 与 --input-dir 不能同时使用")

    cfg = LaserSpotConfig(
        mode=args.mode,
        min_area=args.min_area,
        max_area=args.max_area,
        use_gaussian=not args.no_gaussian,
    )

    if args.input_dir:
        files = list_images_in_dir(args.input_dir)
        if not files:
            raise SystemExit(f"目录中无支持的图像：{args.input_dir}")

        os.makedirs(args.out, exist_ok=True)
        jsonl_path = os.path.join(args.out, "center_result.jsonl")
        if os.path.isfile(jsonl_path):
            os.remove(jsonl_path)

        n = len(files)
        for i, fp in enumerate(files):
            img = imread_unicode(fp)
            rois = select_rois_interactive(
                img,
                index=i + 1,
                total=n,
                basename=os.path.basename(fp),
                full_path=os.path.abspath(fp),
            )
            roi_list = None
            if rois is not None and len(rois) > 0:
                roi_list = rois

            result = locate_laser_spot(
                fp,
                args.out,
                cfg,
                select_roi=False,
                roi_xywh_list=roi_list,
                write_per_image_json=False,
                batch_index=i + 1,
                batch_total=n,
            )
            result["output_jsonl"] = os.path.abspath(jsonl_path)
            append_jsonl_line(jsonl_path, result)
            print(
                f"  [{i + 1}/{n}] {os.path.basename(fp)} -> "
                f"final {result['final_center_xy']}"
            )

        print(f"\n已处理 {n} 张，汇总 JSONL: {os.path.abspath(jsonl_path)}")
        return

    result = locate_laser_spot(
        args.image,
        args.out,
        cfg,
        select_roi=args.select_roi,
        write_per_image_json=True,
        batch_index=1,
        batch_total=1,
    )

    print("\n=== 激光点中心定位结果 ===")
    print(f"灰度加权质心: {result['weighted_center_xy']}")
    print(f"最终中心坐标: {result['final_center_xy']}")
    print(f"Gaussian 信息: {result['gaussian_info']}")
    print(f"结果 JSON: {result.get('output_json')}")
    if result.get("output_final_cross_png"):
        print(f"仅最终十字丝(原图): {result['output_final_cross_png']}")


if __name__ == "__main__":
    main()
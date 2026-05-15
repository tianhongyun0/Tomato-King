import os
import numpy as np
from glob import glob
from typing import Tuple


def _gray_to_binary(gray: np.ndarray) -> np.ndarray:
    """将 Gray Code 整数数组转换为二进制整数（向量化）。"""
    binary = gray.copy()
    mask = gray >> 1
    while np.any(mask):
        binary ^= mask
        mask >>= 1
    return binary


def _expand_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Expand a boolean mask by a small pixel radius without extra dependencies."""
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    padded = np.pad(mask, radius, mode='constant', constant_values=False)
    expanded = np.zeros((h, w), dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            expanded |= padded[
                radius + dy:radius + dy + h,
                radius + dx:radius + dx + w,
            ]
    return expanded


def _continuity_mask(
    proj_x: np.ndarray,
    proj_y: np.ndarray,
    valid_mask: np.ndarray,
    percentile: float = 99.9,
    expand_radius: int = 1,
) -> Tuple[np.ndarray, dict]:
    """
    Reject local jumps in decoded projector coordinates.
    Large jumps are typical of phase unwrap or boundary decode failures.
    """
    valid_mask = valid_mask.astype(bool)
    bad = np.zeros_like(valid_mask, dtype=bool)
    stats = {
        'threshold_x': 0.0,
        'threshold_y': 0.0,
        'bad_pairs_x': 0,
        'bad_pairs_y': 0,
        'removed_pixels': 0,
    }

    if valid_mask.shape[1] > 1:
        dx = np.diff(proj_x, axis=1)
        valid_dx = (
            valid_mask[:, 1:] &
            valid_mask[:, :-1] &
            np.isfinite(dx)
        )
        if valid_dx.any():
            dx_vals = np.abs(dx[valid_dx])
            thr_x = max(4.0, float(np.percentile(dx_vals, percentile)) * 1.5)
            bad_dx = valid_dx & (np.abs(dx) > thr_x)
            if bad_dx.any():
                bad[:, 1:] |= bad_dx
                bad[:, :-1] |= bad_dx
            stats['threshold_x'] = thr_x
            stats['bad_pairs_x'] = int(bad_dx.sum())

    if valid_mask.shape[0] > 1:
        dy = np.diff(proj_y, axis=0)
        valid_dy = (
            valid_mask[1:, :] &
            valid_mask[:-1, :] &
            np.isfinite(dy)
        )
        if valid_dy.any():
            dy_vals = np.abs(dy[valid_dy])
            thr_y = max(4.0, float(np.percentile(dy_vals, percentile)) * 1.5)
            bad_dy = valid_dy & (np.abs(dy) > thr_y)
            if bad_dy.any():
                bad[1:, :] |= bad_dy
                bad[:-1, :] |= bad_dy
            stats['threshold_y'] = thr_y
            stats['bad_pairs_y'] = int(bad_dy.sum())

    bad = _expand_mask(bad, expand_radius)
    stats['removed_pixels'] = int((bad & valid_mask).sum())
    return valid_mask & (~bad), stats


def decode(
    captured_images_dir: str,
    gcp,
    black_threshold: int = 40,
    white_threshold: int = 10,
    proj_w: int = 0,
    proj_h: int = 0,
    skip_high_freq_bits: int = 1,
    camera_matrix=None,
    dist_coeffs=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    解码 Gray Code 图像序列。

    captured_images_dir 中的图像按文件名排序后顺序必须为：
      [编码图案采集图像 x N, 全黑帧采集图像, 全白帧采集图像]
    与 pattern_generator.generate_graycode_patterns() 的输出顺序一致。

    camera_matrix / dist_coeffs:
        相机内参和畸变系数。如果提供，会在解码前对所有图像做去畸变，
        使格雷码坐标系与外参标定坐标系（基于去畸变图像）保持一致。

    返回:
        proj_x_map  [cam_h, cam_w] int32  — 相机像素 → 投影仪 U 坐标，-1 表示无效
        proj_y_map  [cam_h, cam_w] int32  — 相机像素 → 投影仪 V 坐标，-1 表示无效
        valid_mask  [cam_h, cam_w] uint8  — 1=有效, 0=无效
    """
    try:
        import cv2
    except ImportError:
        raise ImportError('需要安装 opencv-contrib-python: pip install opencv-contrib-python')

    image_paths = sorted(
        glob(os.path.join(captured_images_dir, '*.png')) +
        glob(os.path.join(captured_images_dir, '*.jpg')) +
        glob(os.path.join(captured_images_dir, '*.bmp'))
    )
    if not image_paths:
        raise FileNotFoundError(f'在 {captured_images_dir} 中未找到采集图像')

    images = []
    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f'无法读取图像: {p}')
        images.append(img)

    # 如果提供了相机内参和畸变系数，对所有图像做去畸变
    # 使格雷码坐标系与外参标定坐标系（D=0，去畸变图像）保持一致
    if camera_matrix is not None and dist_coeffs is not None:
        K = np.asarray(camera_matrix, dtype=np.float64)
        D = np.asarray(dist_coeffs, dtype=np.float64).flatten()
        if np.any(D != 0):
            images = [cv2.undistort(img, K, D) for img in images]

    images = [img.astype(np.int32) for img in images]

    # 最后两张是全黑/全白（阴影掩膜用），其余是编码图案
    img_black = images[-2]
    img_white = images[-1]
    code_images = images[:-2]

    h, w = images[0].shape

    # 按 pattern_generator 的固定顺序分割：
    # 前 n_col_bits*2 帧是列编码，后 n_row_bits*2 帧是行编码
    import math
    n_col_bits = math.ceil(math.log2(proj_w)) if proj_w else math.ceil(math.log2(max(w, 1)))
    n_row_bits = math.ceil(math.log2(proj_h)) if proj_h else math.ceil(math.log2(max(h, 1)))

    col_frames = code_images[:n_col_bits * 2]
    row_frames = code_images[n_col_bits * 2:]

    # skip_high_freq_bits：跳过最高频的几位（1px、2px条纹在球幕上对比度不足）
    use_col_bits = n_col_bits - skip_high_freq_bits
    use_row_bits = n_row_bits - skip_high_freq_bits
    col_pairs = [(col_frames[i], col_frames[i + 1]) for i in range(0, use_col_bits * 2, 2)]
    row_pairs = [(row_frames[i], row_frames[i + 1]) for i in range(0, use_row_bits * 2, 2)]

    # 阴影掩膜：白-黑差值超过 black_threshold 才认为有有效编码信息
    shadow_mask = (img_white - img_black) > black_threshold

    def decode_axis(pairs, n_bits):
        gray_code = np.zeros((h, w), dtype=np.int32)
        valid = np.ones((h, w), dtype=bool)
        for bit_idx, (pos, neg) in enumerate(pairs):
            diff = pos - neg
            reliable = np.abs(diff) >= white_threshold
            valid &= reliable
            bit = (diff > 0).astype(np.int32)
            gray_code |= (bit << (n_bits - 1 - bit_idx))
        return _gray_to_binary(gray_code), valid

    proj_x_gray, valid_x = decode_axis(col_pairs, use_col_bits)
    proj_y_gray, valid_y = decode_axis(row_pairs, use_row_bits)

    valid = shadow_mask & valid_x & valid_y

    proj_x_map = np.where(valid, proj_x_gray, -1).astype(np.int32)
    proj_y_map = np.where(valid, proj_y_gray, -1).astype(np.int32)
    valid_mask = valid.astype(np.uint8)

    return proj_x_map, proj_y_map, valid_mask


def _load_images(captured_images_dir: str):
    """加载采集图像，返回 int32 图像列表。"""
    try:
        import cv2
    except ImportError:
        raise ImportError('需要安装 opencv-python: pip install opencv-python')

    image_paths = sorted(
        glob(os.path.join(captured_images_dir, '*.png')) +
        glob(os.path.join(captured_images_dir, '*.jpg')) +
        glob(os.path.join(captured_images_dir, '*.bmp'))
    )
    if not image_paths:
        raise FileNotFoundError(f'在 {captured_images_dir} 中未找到采集图像')

    images = []
    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f'无法读取图像: {p}')
        images.append(img.astype(np.int32))
    return images


def decode_hybrid(
    captured_images_dir: str,
    black_threshold: int = 40,
    white_threshold: int = 10,
    proj_w: int = 1920,
    proj_h: int = 1080,
    gc_bits: int = 8,
    phase_quality_threshold: float = 5.0,
    continuity_percentile: float = 99.9,
    continuity_expand_radius: int = 1,
    log=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    格雷码+相位移位混合解码。

    帧布局（与 generate_hybrid_patterns 一致）：
      [0 .. gc_bits*2-1]            列格雷码正负帧
      [gc_bits*2 .. gc_bits*2+3]    列相移4帧
      [gc_bits*2+4 .. gc_bits*4+3]  行格雷码正负帧
      [gc_bits*4+4 .. gc_bits*4+7]  行相移4帧
      [-2]                          全黑帧
      [-1]                          全白帧

    返回:
        proj_x_map  [cam_h, cam_w] float32 — 亚像素 U 坐标（NaN 表示无效）
        proj_y_map  [cam_h, cam_w] float32 — 亚像素 V 坐标（NaN 表示无效）
        valid_mask  [cam_h, cam_w] uint8   — 1=有效, 0=无效
    """
    import math

    images = _load_images(captured_images_dir)
    h, w = images[0].shape

    n_col_bits_full = math.ceil(math.log2(proj_w))
    n_row_bits_full = math.ceil(math.log2(proj_h))
    col_gc_bits = min(gc_bits, n_col_bits_full)
    row_gc_bits = min(gc_bits, n_row_bits_full)

    img_black = images[-2]
    img_white = images[-1]
    frames = images[:-2]

    n_col_gc2 = col_gc_bits * 2
    n_row_gc2 = row_gc_bits * 2

    col_gc_frames = frames[0:n_col_gc2]
    col_phase_frames = frames[n_col_gc2:n_col_gc2 + 4]
    row_gc_start = n_col_gc2 + 4
    row_gc_frames = frames[row_gc_start:row_gc_start + n_row_gc2]
    row_phase_frames = frames[row_gc_start + n_row_gc2:row_gc_start + n_row_gc2 + 4]

    shadow_mask = (img_white - img_black) > black_threshold

    def decode_gc_axis(gc_frames, n_bits):
        pairs = [(gc_frames[i], gc_frames[i + 1]) for i in range(0, n_bits * 2, 2)]
        gray_code = np.zeros((h, w), dtype=np.int32)
        valid = np.ones((h, w), dtype=bool)
        for bit_idx, (pos, neg) in enumerate(pairs):
            diff = pos - neg
            reliable = np.abs(diff) >= white_threshold
            valid &= reliable
            bit = (diff > 0).astype(np.int32)
            gray_code |= (bit << (n_bits - 1 - bit_idx))
        return _gray_to_binary(gray_code), valid

    col_region, valid_col = decode_gc_axis(col_gc_frames, col_gc_bits)
    row_region, valid_row = decode_gc_axis(row_gc_frames, row_gc_bits)

    def decode_phase(phase_frames):
        I0 = phase_frames[0].astype(np.float64)
        I1 = phase_frames[1].astype(np.float64)
        I2 = phase_frames[2].astype(np.float64)
        I3 = phase_frames[3].astype(np.float64)
        phi = np.arctan2(I3 - I1, I0 - I2)
        modulation = 2.0 * np.sqrt((I3 - I1)**2 + (I0 - I2)**2) / 4.0
        return phi, modulation

    col_phi, col_mod = decode_phase(col_phase_frames)
    row_phi, row_mod = decode_phase(row_phase_frames)

    col_period = 1 << (n_col_bits_full - col_gc_bits)
    row_period = 1 << (n_row_bits_full - row_gc_bits)

    def compute_coord(region, phi, period, max_coord):
        offset = ((phi + np.pi) / (2.0 * np.pi) * period) % period - period * 0.5
        coord  = region.astype(np.float64) * period + offset
        region_start = region.astype(np.float64) * period
        coord = np.where(coord < region_start, coord + period, coord)
        coord = np.clip(coord, 0, max_coord - 1)
        return coord.astype(np.float32)

    proj_x = compute_coord(col_region, col_phi, col_period, proj_w)
    proj_y = compute_coord(row_region, row_phi, row_period, proj_h)

    valid_phase = (col_mod >= phase_quality_threshold) & (row_mod >= phase_quality_threshold)
    gc_valid = shadow_mask & valid_col & valid_row & valid_phase

    proj_x_map = np.where(gc_valid, proj_x, np.float32('nan')).astype(np.float32)
    proj_y_map = np.where(gc_valid, proj_y, np.float32('nan')).astype(np.float32)
    continuity_valid, continuity_stats = _continuity_mask(
        proj_x_map,
        proj_y_map,
        gc_valid,
        percentile=continuity_percentile,
        expand_radius=continuity_expand_radius,
    )
    proj_x_map = np.where(continuity_valid, proj_x_map, np.float32('nan')).astype(np.float32)
    proj_y_map = np.where(continuity_valid, proj_y_map, np.float32('nan')).astype(np.float32)
    valid_mask = continuity_valid.astype(np.uint8)

    try:
        import cv2
        x_for_filter = proj_x_map.copy()
        y_for_filter = proj_y_map.copy()
        x_for_filter[valid_mask == 0] = 0
        y_for_filter[valid_mask == 0] = 0
        x_filtered = cv2.medianBlur(cv2.medianBlur(x_for_filter, 5), 5)
        y_filtered = cv2.medianBlur(cv2.medianBlur(y_for_filter, 5), 5)
        proj_x_map = np.where(valid_mask > 0, x_filtered, proj_x_map).astype(np.float32)
        proj_y_map = np.where(valid_mask > 0, y_filtered, proj_y_map).astype(np.float32)
        continuity_valid_2, continuity_stats_2 = _continuity_mask(
            proj_x_map,
            proj_y_map,
            valid_mask > 0,
            percentile=continuity_percentile,
            expand_radius=continuity_expand_radius,
        )
        proj_x_map = np.where(continuity_valid_2, proj_x_map, np.float32('nan')).astype(np.float32)
        proj_y_map = np.where(continuity_valid_2, proj_y_map, np.float32('nan')).astype(np.float32)
        valid_mask = continuity_valid_2.astype(np.uint8)
    except Exception:
        continuity_stats_2 = {
            'threshold_x': 0.0,
            'threshold_y': 0.0,
            'bad_pairs_x': 0,
            'bad_pairs_y': 0,
            'removed_pixels': 0,
        }

    if log is not None:
        total_pixels = valid_mask.size
        shadow_count = int(shadow_mask.sum())
        gc_count = int((shadow_mask & valid_col & valid_row).sum())
        phase_count = int((shadow_mask & valid_col & valid_row & valid_phase).sum())
        final_count = int(valid_mask.sum())
        log(
            "  hybrid decode stats: "
            f"shadow={shadow_count}/{total_pixels}, "
            f"phase_reject={gc_count - phase_count}, "
            f"continuity_reject_1={continuity_stats['removed_pixels']}, "
            f"continuity_reject_2={continuity_stats_2['removed_pixels']}, "
            f"final={final_count}/{total_pixels}"
        )

    return proj_x_map, proj_y_map, valid_mask

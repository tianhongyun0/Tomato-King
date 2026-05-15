import csv
import json
import argparse
from pathlib import Path

import numpy as np


# ============================================================
# 基础工具
# ============================================================

def dms_to_deg(deg, minutes, sec):
    """
    度分秒转十进制度。
    用字符串处理 deg，是为了保留 "-0" 这种情况。
    """
    deg_str = str(deg).strip()
    sign = -1 if deg_str.startswith("-") else 1

    deg_abs = abs(float(deg_str))
    minutes = float(minutes)
    sec = float(sec)

    return sign * (deg_abs + minutes / 60.0 + sec / 3600.0)


def measurement_to_unit_vector(az, el):
    """
    全站仪方向角转单位射线方向。

    坐标系：
      X = d * cos(el) * sin(az)
      Y = d * cos(el) * cos(az)
      Z = d * sin(el)
    """
    return np.array([
        np.cos(el) * np.sin(az),
        np.cos(el) * np.cos(az),
        np.sin(el),
    ], dtype=float)


def load_measurements_from_csv(csv_path, enabled_only=True, exclude_point_ids=None):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到输入 CSV 文件：{csv_path}")

    exclude_point_ids = set(exclude_point_ids or [])
    measurements = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required_columns = [
            "h_deg", "h_min", "h_sec",
            "v_deg", "v_min", "v_sec",
            "distance_m",
        ]

        missing = [c for c in required_columns if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV 缺少必要列：{missing}")

        for row_index, row in enumerate(reader, start=2):
            enabled_value = str(row.get("enabled", "1")).strip()
            point_id = row.get("point_id", f"row_{row_index}")

            if enabled_only and enabled_value not in ("1", "true", "True", "TRUE", "yes", "YES"):
                continue

            if point_id in exclude_point_ids:
                continue

            distance_m = float(row["distance_m"])
            distance_mm = distance_m * 1000.0

            h_deg_decimal = dms_to_deg(row["h_deg"], row["h_min"], row["h_sec"])
            v_deg_decimal = dms_to_deg(row["v_deg"], row["v_min"], row["v_sec"])

            h_rad = np.radians(h_deg_decimal)
            v_rad = np.radians(v_deg_decimal)

            measurements.append({
                "row_index": row_index,
                "image_name": row.get("image_name", ""),
                "point_id": point_id,

                "h_deg": row["h_deg"],
                "h_min": row["h_min"],
                "h_sec": row["h_sec"],

                "v_deg": row["v_deg"],
                "v_min": row["v_min"],
                "v_sec": row["v_sec"],

                "h_deg_decimal": h_deg_decimal,
                "v_deg_decimal": v_deg_decimal,

                "h_rad": h_rad,
                "v_rad": v_rad,

                "distance_m": distance_m,
                "distance_mm": distance_mm,
            })

    if len(measurements) < 4:
        raise ValueError(
            f"有效点数量不足：{len(measurements)}。拟合 3D 球面至少需要 4 个非共面的点。"
        )

    return measurements


def measurements_to_xyz_and_rays(measurements):
    pts = []
    rays = []
    dists = []

    for m in measurements:
        az = float(m["h_rad"])
        el = float(m["v_rad"])
        d = float(m["distance_mm"])

        u = measurement_to_unit_vector(az, el)
        p = d * u

        rays.append(u)
        pts.append(p)
        dists.append(d)

    return (
        np.array(pts, dtype=float),
        np.array(rays, dtype=float),
        np.array(dists, dtype=float),
    )


# ============================================================
# 全站仪误差传播：用于加权几何最小二乘
# ============================================================

def point_covariance_from_total_station(
    az,
    el,
    d_mm,
    sigma_h_rad,
    sigma_v_rad,
    sigma_distance_mm,
):
    """
    把全站仪水平角、竖直角与斜距误差传播到 XYZ 点协方差 Σ_P。

    P = d * u(az, el)。观测方差取对角：
      diag(σ_h², σ_v², σ_d²)，分别对应水平方向角、竖直角、斜距（独立假设）。
    """
    sin_az = np.sin(az)
    cos_az = np.cos(az)
    sin_el = np.sin(el)
    cos_el = np.cos(el)

    # dP / d az（水平角）
    dp_daz = d_mm * np.array([
        cos_el * cos_az,
        -cos_el * sin_az,
        0.0,
    ])

    # dP / d el（竖直角）
    dp_del = d_mm * np.array([
        -sin_el * sin_az,
        -sin_el * cos_az,
        cos_el,
    ])

    # dP / d d（斜距）
    dp_dd = np.array([
        cos_el * sin_az,
        cos_el * cos_az,
        sin_el,
    ])

    J = np.column_stack([dp_daz, dp_del, dp_dd])

    sigma_obs = np.diag([
        float(sigma_h_rad) ** 2,
        float(sigma_v_rad) ** 2,
        float(sigma_distance_mm) ** 2,
    ])

    cov_p = J @ sigma_obs @ J.T

    return cov_p


def build_point_covariances(
    measurements,
    sigma_h_arcsec,
    sigma_distance_mm,
    sigma_v_arcsec=None,
):
    """
    为每个测回构建 3×3 协方差。若 sigma_v_arcsec 为 None，则竖直角标准差与水平角相同。
    """
    sigma_h_rad = np.deg2rad(float(sigma_h_arcsec) / 3600.0)
    if sigma_v_arcsec is None:
        sigma_v_rad = sigma_h_rad
    else:
        sigma_v_rad = np.deg2rad(float(sigma_v_arcsec) / 3600.0)

    covs = []

    for m in measurements:
        cov = point_covariance_from_total_station(
            az=float(m["h_rad"]),
            el=float(m["v_rad"]),
            d_mm=float(m["distance_mm"]),
            sigma_h_rad=sigma_h_rad,
            sigma_v_rad=sigma_v_rad,
            sigma_distance_mm=float(sigma_distance_mm),
        )
        covs.append(cov)

    return np.array(covs, dtype=float)


def ray_mahalanobis_weight_squared(u, cov):
    """
    计算 u^T Σ^{-1} u（u 为单位方向）。

    用于「观测点 P = d·u」在误差近似沿射线方向 (d - t)·u 时，
    将 (d - t) 按马氏距离归一化，使角度与斜距在 Σ 中一并参与 directional 目标。
    """
    u = np.asarray(u, dtype=float).ravel()
    cov = np.asarray(cov, dtype=float)

    try:
        L = np.linalg.cholesky(cov)
        w = np.linalg.solve(L, u)
        return float(np.dot(w, w))
    except np.linalg.LinAlgError:
        sol, _, _, _ = np.linalg.lstsq(cov, u, rcond=None)
        return float(np.dot(u, sol))


# ============================================================
# 方法 1：代数最小二乘球拟合
# ============================================================

def fit_sphere_algebraic(pts):
    """
    代数最小二乘球拟合。

    球面：
      (x - cx)^2 + (y - cy)^2 + (z - cz)^2 = R^2

    展开：
      2cx*x + 2cy*y + 2cz*z + b = x^2 + y^2 + z^2

    其中：
      b = R^2 - cx^2 - cy^2 - cz^2
    """
    A = np.column_stack([
        2.0 * pts[:, 0],
        2.0 * pts[:, 1],
        2.0 * pts[:, 2],
        np.ones(len(pts)),
    ])

    rhs = np.sum(pts ** 2, axis=1)

    solution, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)

    cx, cy, cz, b = solution

    radius_sq = b + cx ** 2 + cy ** 2 + cz ** 2

    if radius_sq <= 0:
        raise ValueError(f"代数球拟合失败，半径平方异常：{radius_sq}")

    center = np.array([cx, cy, cz], dtype=float)
    radius = float(np.sqrt(radius_sq))

    return center, radius, {
        "solver": "numpy.linalg.lstsq",
        "iterations": 1,
    }


# ============================================================
# 方法 2：几何非线性最小二乘
# ============================================================

def fit_sphere_geometric_ls(pts, center0=None, radius0=None):
    """
    几何非线性最小二乘。

    最小化：
      sum_i ( ||P_i - C|| - R )^2
    """
    try:
        from scipy.optimize import least_squares
    except ImportError as e:
        raise ImportError("需要 scipy。请先运行：pip install scipy") from e

    if center0 is None or radius0 is None:
        center0, radius0, _ = fit_sphere_algebraic(pts)

    x0 = np.array([
        center0[0],
        center0[1],
        center0[2],
        radius0,
    ], dtype=float)

    def residual_func(params):
        cx, cy, cz, radius = params

        if radius <= 0:
            return np.ones(len(pts), dtype=float) * 1e6

        center = np.array([cx, cy, cz], dtype=float)
        dists = np.linalg.norm(pts - center, axis=1)

        return dists - radius

    result = least_squares(
        residual_func,
        x0,
        method="trf",
        bounds=(
            [-np.inf, -np.inf, -np.inf, 1e-9],
            [np.inf, np.inf, np.inf, np.inf],
        ),
        max_nfev=10000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )

    if not result.success:
        raise RuntimeError(f"geometric-ls 拟合失败：{result.message}")

    cx, cy, cz, radius = result.x
    center = np.array([cx, cy, cz], dtype=float)

    return center, float(radius), {
        "solver": "scipy.optimize.least_squares",
        "success": bool(result.success),
        "message": result.message,
        "iterations": int(result.nfev),
        "cost": float(result.cost),
    }


def _pack_sphere_log_radius(center, radius):
    r = max(float(radius), 1e-12)
    return np.array(
        [float(center[0]), float(center[1]), float(center[2]), np.log(r)],
        dtype=float,
    )


def _unpack_sphere_log_radius(x):
    return np.array(x[:3], dtype=float), float(np.exp(x[3]))


def fit_sphere_geometric_lm(pts, center0=None, radius0=None):
    """
    Levenberg–Marquardt（scipy least_squares method=lm）几何球拟合。

    半径用 log(R) 参数化，避免 LM 不支持 R 下界时半径变负。
    提供解析雅可比，对噪声初值通常比纯 TRF 阻尼牛顿更稳。
    """
    try:
        from scipy.optimize import least_squares
    except ImportError as e:
        raise ImportError("需要 scipy。请先运行：pip install scipy") from e

    if center0 is None or radius0 is None:
        center0, radius0, _ = fit_sphere_algebraic(pts)

    x0 = _pack_sphere_log_radius(center0, radius0)

    def residual_func(x):
        center, radius = _unpack_sphere_log_radius(x)
        if radius <= 0:
            return np.ones(len(pts), dtype=float) * 1e6
        dists = np.linalg.norm(pts - center, axis=1)
        return dists - radius

    def jac_func(x):
        center, radius = _unpack_sphere_log_radius(x)
        vecs = pts - center
        dists = np.linalg.norm(vecs, axis=1)
        dists = np.maximum(dists, 1e-12)
        J = np.zeros((len(pts), 4), dtype=float)
        J[:, 0] = -vecs[:, 0] / dists
        J[:, 1] = -vecs[:, 1] / dists
        J[:, 2] = -vecs[:, 2] / dists
        J[:, 3] = -radius
        return J

    result = least_squares(
        residual_func,
        x0,
        jac=jac_func,
        method="lm",
        max_nfev=10000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )

    if not result.success:
        raise RuntimeError(f"geometric-lm 拟合失败：{result.message}")

    center, radius = _unpack_sphere_log_radius(result.x)

    return center, float(radius), {
        "solver": "scipy.optimize.least_squares_lm",
        "success": bool(result.success),
        "message": result.message,
        "iterations": int(result.nfev),
        "cost": float(result.cost),
    }


# ============================================================
# 方法 3：加权几何最小二乘
# ============================================================

def fit_sphere_weighted_geometric_ls(pts, point_covs, center0=None, radius0=None):
    """
    加权几何最小二乘。

    几何残差：
      e_i = ||P_i - C|| - R

    点位协方差投影到球面法向：
      sigma_e_i^2 = n_i^T Sigma_P_i n_i

    最小化：
      sum_i (e_i / sigma_e_i)^2
    """
    try:
        from scipy.optimize import least_squares
    except ImportError as e:
        raise ImportError("需要 scipy。请先运行：pip install scipy") from e

    if center0 is None or radius0 is None:
        center0, radius0, _ = fit_sphere_algebraic(pts)

    x0 = np.array([
        center0[0],
        center0[1],
        center0[2],
        radius0,
    ], dtype=float)

    def residual_func(params):
        cx, cy, cz, radius = params

        if radius <= 0:
            return np.ones(len(pts), dtype=float) * 1e6

        center = np.array([cx, cy, cz], dtype=float)

        vecs = pts - center
        dists = np.linalg.norm(vecs, axis=1)
        dists = np.maximum(dists, 1e-12)

        normals = vecs / dists[:, None]
        geom_residuals = dists - radius

        weighted_residuals = []

        for i in range(len(pts)):
            n = normals[i]
            cov = point_covs[i]

            var_e = float(n.T @ cov @ n)
            var_e = max(var_e, 1e-12)

            weighted_residuals.append(geom_residuals[i] / np.sqrt(var_e))

        return np.array(weighted_residuals, dtype=float)

    result = least_squares(
        residual_func,
        x0,
        method="trf",
        bounds=(
            [-np.inf, -np.inf, -np.inf, 1e-9],
            [np.inf, np.inf, np.inf, np.inf],
        ),
        max_nfev=10000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )

    if not result.success:
        raise RuntimeError(f"weighted-geometric-ls 拟合失败：{result.message}")

    cx, cy, cz, radius = result.x
    center = np.array([cx, cy, cz], dtype=float)

    return center, float(radius), {
        "solver": "scipy.optimize.least_squares_weighted",
        "success": bool(result.success),
        "message": result.message,
        "iterations": int(result.nfev),
        "cost": float(result.cost),
    }


def fit_sphere_weighted_geometric_lm(pts, point_covs, center0=None, radius0=None):
    """
    加权几何最小二乘 + Levenberg–Marquardt（scipy method=lm，log R 参数化）。

    权重随球心变化；对雅可比采用中心差分（2-point），适合协方差投影权。
    """
    try:
        from scipy.optimize import least_squares
    except ImportError as e:
        raise ImportError("需要 scipy。请先运行：pip install scipy") from e

    if center0 is None or radius0 is None:
        center0, radius0, _ = fit_sphere_algebraic(pts)

    x0 = _pack_sphere_log_radius(center0, radius0)

    def residual_func(x):
        center, radius = _unpack_sphere_log_radius(x)
        if radius <= 0:
            return np.ones(len(pts), dtype=float) * 1e6

        vecs = pts - center
        dists = np.linalg.norm(vecs, axis=1)
        dists = np.maximum(dists, 1e-12)
        normals = vecs / dists[:, None]
        geom_residuals = dists - radius

        weighted_residuals = []
        for i in range(len(pts)):
            n = normals[i]
            cov = point_covs[i]
            var_e = float(n.T @ cov @ n)
            var_e = max(var_e, 1e-12)
            weighted_residuals.append(geom_residuals[i] / np.sqrt(var_e))

        return np.array(weighted_residuals, dtype=float)

    result = least_squares(
        residual_func,
        x0,
        jac="2-point",
        method="lm",
        max_nfev=10000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )

    if not result.success:
        raise RuntimeError(f"weighted-geometric-lm 拟合失败：{result.message}")

    center, radius = _unpack_sphere_log_radius(result.x)

    return center, float(radius), {
        "solver": "scipy.optimize.least_squares_weighted_lm_2point_jac",
        "success": bool(result.success),
        "message": result.message,
        "iterations": int(result.nfev),
        "cost": float(result.cost),
    }


def _tangent_basis_perp_to_unit(w):
    """返回 3×2 矩阵 H，列正交单位且与单位向量 w 正交。"""
    w = np.asarray(w, dtype=float).ravel()
    w = w / max(np.linalg.norm(w), 1e-15)
    if abs(w[2]) < 0.9:
        e1 = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        e1 = np.array([1.0, 0.0, 0.0], dtype=float)
    h1 = e1 - float(np.dot(e1, w)) * w
    h1 = h1 / max(np.linalg.norm(h1), 1e-15)
    h2 = np.cross(w, h1)
    h2 = h2 / max(np.linalg.norm(h2), 1e-15)
    return np.column_stack([h1, h2])


def whitened_residual_mahal_foot_one_point(P, center, radius, cov):
    """
    马氏意义下球上垂足的一步切平面近似，返回 Cholesky 白化后的 3 维残差。

    先用欧氏垂足 w0，再在 w0 切平面上解 2×2 线性方程修正 w，使
    (P - S)^T Σ^{-1} (P - S) 近似最小，S = C + R w，||w||=1。
    """
    P = np.asarray(P, dtype=float).ravel()
    center = np.asarray(center, dtype=float).ravel()
    cov = np.asarray(cov, dtype=float)

    b = P - center
    dn = float(np.linalg.norm(b))
    if dn < 1e-12:
        return np.ones(3, dtype=float) * 1e3

    w0 = b / dn
    delta0 = P - (center + radius * w0)

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(cov + np.eye(3) * 1e-9)

    H = _tangent_basis_perp_to_unit(w0)

    K = H.T @ np.linalg.solve(cov, H)
    rhs = H.T @ np.linalg.solve(cov, delta0)

    try:
        eta = (1.0 / max(radius, 1e-12)) * np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        eta = np.zeros(2, dtype=float)

    w = w0 + H @ eta
    w = w / max(np.linalg.norm(w), 1e-15)
    S = center + radius * w
    diff = P - S

    return np.linalg.solve(L, diff)


def fit_sphere_mahal_orthogonal_ls(pts, point_covs, center0=None, radius0=None):
    """
    马氏正交距离（近似）：每个观测点用 Σ_i^{-1} 度量到球面的偏差，
    球上垂足用欧氏垂足 + 一步切平面修正近似。

    残差为 3×N 维（白化后的 P-S），比单标量法向投影多利用协方差的切向信息。
    """
    try:
        from scipy.optimize import least_squares
    except ImportError as e:
        raise ImportError("需要 scipy。请先运行：pip install scipy") from e

    if center0 is None or radius0 is None:
        center0, radius0, _ = fit_sphere_algebraic(pts)

    x0 = np.array([
        center0[0],
        center0[1],
        center0[2],
        radius0,
    ], dtype=float)

    def residual_func(params):
        cx, cy, cz, radius = params

        if radius <= 0:
            return np.ones(len(pts) * 3, dtype=float) * 1e6

        center = np.array([cx, cy, cz], dtype=float)
        out = []

        for i in range(len(pts)):
            r3 = whitened_residual_mahal_foot_one_point(
                pts[i], center, radius, point_covs[i]
            )
            out.extend(r3.tolist())

        return np.array(out, dtype=float)

    result = least_squares(
        residual_func,
        x0,
        method="trf",
        bounds=(
            [-np.inf, -np.inf, -np.inf, 1e-9],
            [np.inf, np.inf, np.inf, np.inf],
        ),
        max_nfev=10000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )

    if not result.success:
        raise RuntimeError(f"mahal-orthogonal-ls 拟合失败：{result.message}")

    cx, cy, cz, radius = result.x
    center = np.array([cx, cy, cz], dtype=float)

    return center, float(radius), {
        "solver": "scipy.optimize.least_squares_mahal_orthogonal",
        "success": bool(result.success),
        "message": result.message,
        "iterations": int(result.nfev),
        "cost": float(result.cost),
    }


# ============================================================
# 方法 4：角度 + 斜距观测域拟合
# ============================================================

def ray_sphere_intersection_distance(ray_u, center, radius, measured_distance_mm):
    """
    射线：
      P(t) = t * u

    球：
      ||P - C||^2 = R^2

    解：
      t^2 - 2(u·C)t + ||C||^2 - R^2 = 0

    返回最接近 measured_distance_mm 的正根。
    """
    u_dot_c = float(ray_u @ center)
    c_dot_c = float(center @ center)

    discriminant = u_dot_c ** 2 - (c_dot_c - radius ** 2)

    if discriminant < 0:
        return None

    sqrt_disc = np.sqrt(discriminant)

    t1 = u_dot_c - sqrt_disc
    t2 = u_dot_c + sqrt_disc

    candidates = [t for t in (t1, t2) if t > 0]

    if not candidates:
        return None

    return min(candidates, key=lambda t: abs(t - measured_distance_mm))


def fit_sphere_directional_ls(
    rays,
    measured_distances_mm,
    center0=None,
    radius0=None,
    pts_for_initial=None,
    sigma_distance_mm=1.0,
    point_covs=None,
    directional_weight="ray-mahalanobis",
):
    """
    角度 + 斜距观测域拟合。

    对每个点：
      已知射线方向 u_i
      已知斜距 d_i

    给定球心 C 和半径 R，射线与球的理论交点距离为 t_i。

    directional_weight == "distance-only"（与旧版一致）：
      最小化 sum_i ((d_i - t_i(C, R)) / sigma_d)^2

    directional_weight == "ray-mahalanobis"（默认，需传入 point_covs）：
      将笛卡尔误差近似为 (d_i - t_i)·u_i，用点位协方差 Σ_i 的马氏范数归一：
      sum_i (d_i - t_i)^2 * (u_i^T Σ_i^{-1} u_i)
      其中 Σ_i 由角度与斜距方差经全站仪误差传播得到（与 weighted-geometric-ls 同源）。
    """
    try:
        from scipy.optimize import least_squares
    except ImportError as e:
        raise ImportError("需要 scipy。请先运行：pip install scipy") from e

    if center0 is None or radius0 is None:
        if pts_for_initial is None:
            pts_for_initial = measured_distances_mm[:, None] * rays
        center0, radius0, _ = fit_sphere_algebraic(pts_for_initial)

    x0 = np.array([
        center0[0],
        center0[1],
        center0[2],
        radius0,
    ], dtype=float)

    sigma_distance_mm = max(float(sigma_distance_mm), 1e-12)
    directional_weight = str(directional_weight or "ray-mahalanobis").strip().lower()

    if directional_weight not in ("ray-mahalanobis", "distance-only"):
        raise ValueError(
            f"directional_weight 必须是 ray-mahalanobis 或 distance-only，收到：{directional_weight!r}"
        )

    if directional_weight == "ray-mahalanobis" and point_covs is None:
        raise ValueError(
            "directional_weight=ray-mahalanobis 时必须提供 point_covs（与 CSV 中每点一一对应）。"
        )

    def residual_func(params):
        cx, cy, cz, radius = params

        if radius <= 0:
            return np.ones(len(rays), dtype=float) * 1e6

        center = np.array([cx, cy, cz], dtype=float)

        residuals = []

        for i, (u, measured_d) in enumerate(zip(rays, measured_distances_mm)):
            t_pred = ray_sphere_intersection_distance(
                ray_u=u,
                center=center,
                radius=radius,
                measured_distance_mm=measured_d,
            )

            if t_pred is None:
                residuals.append(1e6)
            else:
                delta = float(measured_d - t_pred)
                if directional_weight == "distance-only":
                    residuals.append(delta / sigma_distance_mm)
                else:
                    w2 = ray_mahalanobis_weight_squared(u, point_covs[i])
                    w2 = max(w2, 1e-24)
                    residuals.append(delta * np.sqrt(w2))

        return np.array(residuals, dtype=float)

    result = least_squares(
        residual_func,
        x0,
        method="trf",
        bounds=(
            [-np.inf, -np.inf, -np.inf, 1e-9],
            [np.inf, np.inf, np.inf, np.inf],
        ),
        max_nfev=10000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )

    if not result.success:
        raise RuntimeError(f"directional-ls 拟合失败：{result.message}")

    cx, cy, cz, radius = result.x
    center = np.array([cx, cy, cz], dtype=float)

    return center, float(radius), {
        "solver": "scipy.optimize.least_squares_directional",
        "success": bool(result.success),
        "message": result.message,
        "iterations": int(result.nfev),
        "cost": float(result.cost),
        "directional_weight": directional_weight,
    }


# ============================================================
# 方法 5：手写 Gauss-Newton 几何球拟合
# ============================================================

def geometric_residual_and_jacobian(pts, params):
    """
    几何残差和雅可比。

    参数：
      params = [cx, cy, cz, R]

    残差：
      r_i = ||P_i - C|| - R

    雅可比：
      dr_i / dcx = -(x_i - cx) / ||P_i - C||
      dr_i / dcy = -(y_i - cy) / ||P_i - C||
      dr_i / dcz = -(z_i - cz) / ||P_i - C||
      dr_i / dR  = -1
    """
    cx, cy, cz, radius = params
    center = np.array([cx, cy, cz], dtype=float)

    vecs = pts - center
    dists = np.linalg.norm(vecs, axis=1)
    dists = np.maximum(dists, 1e-12)

    residuals = dists - radius

    J = np.zeros((len(pts), 4), dtype=float)
    J[:, 0] = -vecs[:, 0] / dists
    J[:, 1] = -vecs[:, 1] / dists
    J[:, 2] = -vecs[:, 2] / dists
    J[:, 3] = -1.0

    return residuals, J


def fit_sphere_gauss_newton_geometric(
    pts,
    center0=None,
    radius0=None,
    max_iter=100,
    tol=1e-12,
    damping=1e-8,
):
    """
    手写 Gauss-Newton 几何球拟合。

    目标：
      min sum_i (||P_i - C|| - R)^2

    说明：
      这里加了很小的 damping，避免 J^T J 奇异或病态。
      damping=0 时更接近纯 Gauss-Newton。
    """
    if center0 is None or radius0 is None:
        center0, radius0, _ = fit_sphere_algebraic(pts)

    x = np.array([
        center0[0],
        center0[1],
        center0[2],
        radius0,
    ], dtype=float)

    last_cost = None
    actual_iter = 0

    for it in range(max_iter):
        actual_iter = it + 1

        residuals, J = geometric_residual_and_jacobian(pts, x)

        cost = float(np.sum(residuals ** 2))

        A = J.T @ J
        g = J.T @ residuals

        if damping > 0:
            A = A + damping * np.eye(4)

        try:
            delta = np.linalg.solve(A, -g)
        except np.linalg.LinAlgError:
            delta, _, _, _ = np.linalg.lstsq(A, -g, rcond=None)

        # 简单线搜索，防止半径变成负数或 cost 变大太多
        alpha = 1.0
        accepted = False

        for _ in range(20):
            x_try = x + alpha * delta

            if x_try[3] <= 0:
                alpha *= 0.5
                continue

            r_try, _ = geometric_residual_and_jacobian(pts, x_try)
            cost_try = float(np.sum(r_try ** 2))

            if cost_try <= cost:
                x = x_try
                accepted = True
                break

            alpha *= 0.5

        if not accepted:
            x = x + delta

        step_norm = float(np.linalg.norm(alpha * delta))

        if last_cost is not None:
            cost_change = abs(last_cost - cost)
        else:
            cost_change = np.inf

        last_cost = cost

        if step_norm < tol or cost_change < tol:
            break

    cx, cy, cz, radius = x
    center = np.array([cx, cy, cz], dtype=float)

    return center, float(radius), {
        "solver": "manual_gauss_newton",
        "iterations": actual_iter,
        "damping": float(damping),
        "final_cost": float(last_cost if last_cost is not None else 0.0),
    }


def _weighted_geometric_residual_jacobian_frozen(pts, center, radius, point_covs):
    """
    加权几何残差 r_i = w_i e_i，其中 e_i = ||P_i-C||-R，
    w_i = 1/sqrt(n_i^T Σ_i n_i)，n_i 与 σ_i 均在当前 (C,R) 处取值（一步内冻结）。
    """
    center = np.asarray(center, dtype=float).ravel()
    vecs = pts - center
    dists = np.linalg.norm(vecs, axis=1)
    dists = np.maximum(dists, 1e-12)
    normals = vecs / dists[:, None]
    e = dists - float(radius)
    npt = len(pts)
    r = np.zeros(npt, dtype=float)
    J = np.zeros((npt, 4), dtype=float)
    for i in range(npt):
        n = normals[i]
        var_e = float(n.T @ point_covs[i] @ n)
        w = 1.0 / np.sqrt(max(var_e, 1e-12))
        r[i] = w * e[i]
        J[i, 0] = w * (-n[0])
        J[i, 1] = w * (-n[1])
        J[i, 2] = w * (-n[2])
        J[i, 3] = w * (-1.0)
    cost = float(np.dot(r, r))
    return r, J, cost


def fit_sphere_weighted_marquardt_geometric(
    pts,
    point_covs,
    center0=None,
    radius0=None,
    max_iter=200,
    lambda_init=1e-3,
    tol=1e-12,
):
    """
    加权几何最小二乘 + Levenberg–Marquardt 阻尼（手写 Marquardt 对角缩放）。

    每步解 (J^T J + λ diag(J^T J)) δ = -J^T r；成功则减小 λ，失败则增大 λ。
    雅可比在**当前**球心下用法向方差 σ_i² = n_i^T Σ_i n_i 冻结权重（与 IRLS 一步一致）。
    """
    if center0 is None or radius0 is None:
        center0, radius0, _ = fit_sphere_algebraic(pts)

    x = np.array([
        center0[0],
        center0[1],
        center0[2],
        max(float(radius0), 1e-12),
    ], dtype=float)

    lam = max(float(lambda_init), 1e-15)
    actual_iter = 0

    for it in range(max_iter):
        actual_iter = it + 1
        center = x[:3]
        radius = x[3]
        r, J, cost = _weighted_geometric_residual_jacobian_frozen(
            pts, center, radius, point_covs
        )

        JTJ = J.T @ J
        g = J.T @ r
        d_scale = np.diag(JTJ).copy()
        d_scale = np.maximum(d_scale, 1e-18)

        step_accepted = False
        for _ in range(40):
            A = JTJ + lam * np.diag(d_scale)
            try:
                delta = np.linalg.solve(A, -g)
            except np.linalg.LinAlgError:
                delta, _, _, _ = np.linalg.lstsq(A, -g, rcond=None)

            x_try = x + delta
            if x_try[3] <= 0:
                x_try[3] = 1e-9

            _, _, cost_try = _weighted_geometric_residual_jacobian_frozen(
                pts, x_try[:3], x_try[3], point_covs
            )

            if cost_try < cost:
                x = x_try
                lam = max(lam * 0.2, 1e-15)
                step_accepted = True
                break

            lam = min(lam * 5.0, 1e20)

        if not step_accepted:
            break

        if float(np.linalg.norm(delta)) < tol:
            break

    _, _, final_cost = _weighted_geometric_residual_jacobian_frozen(
        pts, x[:3], x[3], point_covs
    )

    cx, cy, cz, radius = x[0], x[1], x[2], x[3]
    center = np.array([cx, cy, cz], dtype=float)

    return center, float(radius), {
        "solver": "manual_levenberg_marquardt_weighted_geometric",
        "iterations": actual_iter,
        "final_lambda": float(lam),
        "final_cost": float(final_cost),
    }


def fit_sphere_weighted_sequential_ekf(
    pts,
    point_covs,
    center0=None,
    radius0=None,
    max_sweeps=50,
    p_std_mm=(1e3, 1e3, 1e3, 100.0),
    tol=1e-10,
):
    """
    迭代加权高斯–牛顿（法向观测信息累积，与「顺序 IEKF / 批量加权非线性 LS」同构）。

    将每个点的几何残差 h_i = ||P_i-C||-R 视为伪观测，方差 R_i = n_i^T Σ_i n_i。
    每轮扫描在**当前** θ 处线性化，累积
      A ≈ Σ_i (1/R_i) H_i^T H_i，  b = Σ_i (1/R_i) H_i^T h_i，
    解 A δ = -b 更新 θ，直至 ‖δ‖ 小于容差。

    说明：经典「逐点 Joseph EKF」对非线性球面模型与观测顺序强相关，易偏离批量加权 LS；
    这里采用测量域协方差 R_i 下的**迭代线性化 GN**，与加权几何最小二乘目标一致、数值更稳。
    """
    _ = p_std_mm  # 保留 API；当前为协方差加权下的迭代 GN，不使用对角先验 P0

    if center0 is None or radius0 is None:
        center0, radius0, _ = fit_sphere_algebraic(pts)

    theta = np.array([
        center0[0],
        center0[1],
        center0[2],
        max(float(radius0), 1e-12),
    ], dtype=float)

    sweeps_done = 0
    for sweep in range(int(max_sweeps)):
        sweeps_done = sweep + 1
        A = np.zeros((4, 4), dtype=float)
        bvec = np.zeros(4, dtype=float)

        for i in range(len(pts)):
            C = theta[:3]
            R = max(float(theta[3]), 1e-12)
            vecs = pts[i] - C
            d = float(np.linalg.norm(vecs))
            if d < 1e-12:
                continue
            nvec = vecs / d
            h = d - R
            R_meas = float(nvec @ point_covs[i] @ nvec)
            R_meas = max(R_meas, 1e-12)

            Hi = np.zeros(4, dtype=float)
            Hi[:3] = -nvec
            Hi[3] = -1.0

            A += np.outer(Hi, Hi) / R_meas
            bvec += Hi * (h / R_meas)

        A = A + np.eye(4, dtype=float) * 1e-12
        try:
            delta = np.linalg.solve(A, -bvec)
        except np.linalg.LinAlgError:
            delta, _, _, _ = np.linalg.lstsq(A, -bvec, rcond=None)

        theta = theta + delta
        theta[3] = max(theta[3], 1e-12)

        if float(np.linalg.norm(delta)) < tol:
            break

    center = np.array(theta[:3], dtype=float)
    radius = float(theta[3])

    return center, radius, {
        "solver": "iterative_weighted_gauss_newton_normal_eqs",
        "sweeps": int(sweeps_done),
        "note": "与加权几何 LS 同一目标；等价于协方差加权下的迭代线性化卡尔曼/IEKF 批量步",
    }


# ============================================================
# 残差计算
# ============================================================

def compute_geometric_residuals(pts, center, radius):
    """
    点到球面的几何残差：
      ||P_i - C|| - R
    """
    dists = np.linalg.norm(pts - center, axis=1)
    residuals = dists - radius

    rms = float(np.sqrt(np.mean(residuals ** 2)))
    max_abs = float(np.max(np.abs(residuals)))

    return residuals, rms, max_abs


def compute_directional_residuals(rays, measured_distances_mm, center, radius):
    """
    全站仪射线方向残差：
      measured_distance - predicted_intersection_distance
    """
    residuals = []

    for u, measured_d in zip(rays, measured_distances_mm):
        t_pred = ray_sphere_intersection_distance(
            ray_u=u,
            center=center,
            radius=radius,
            measured_distance_mm=measured_d,
        )

        if t_pred is None:
            residuals.append(float("nan"))
        else:
            residuals.append(float(measured_d - t_pred))

    residuals = np.array(residuals, dtype=float)
    finite = np.isfinite(residuals)

    if np.any(finite):
        rms = float(np.sqrt(np.mean(residuals[finite] ** 2)))
        max_abs = float(np.max(np.abs(residuals[finite])))
    else:
        rms = float("nan")
        max_abs = float("nan")

    return residuals, rms, max_abs


# ============================================================
# 拟合统一入口
# ============================================================

def fit_by_method(
    method,
    pts,
    rays,
    measured_distances_mm,
    point_covs,
    sigma_distance_mm,
    gn_max_iter,
    gn_damping,
    directional_weight="ray-mahalanobis",
    marquardt_max_iter=200,
    marquardt_lambda_init=1e-3,
    ekf_max_sweeps=50,
):
    center_alg, radius_alg, _ = fit_sphere_algebraic(pts)

    if method == "algebraic":
        return fit_sphere_algebraic(pts)

    if method == "geometric-ls":
        return fit_sphere_geometric_ls(
            pts=pts,
            center0=center_alg,
            radius0=radius_alg,
        )

    if method == "geometric-lm":
        return fit_sphere_geometric_lm(
            pts=pts,
            center0=center_alg,
            radius0=radius_alg,
        )

    if method == "weighted-geometric-ls":
        return fit_sphere_weighted_geometric_ls(
            pts=pts,
            point_covs=point_covs,
            center0=center_alg,
            radius0=radius_alg,
        )

    if method == "weighted-geometric-lm":
        return fit_sphere_weighted_geometric_lm(
            pts=pts,
            point_covs=point_covs,
            center0=center_alg,
            radius0=radius_alg,
        )

    if method == "weighted-marquardt-geometric":
        return fit_sphere_weighted_marquardt_geometric(
            pts=pts,
            point_covs=point_covs,
            center0=center_alg,
            radius0=radius_alg,
            max_iter=marquardt_max_iter,
            lambda_init=marquardt_lambda_init,
        )

    if method == "weighted-sequential-ekf":
        return fit_sphere_weighted_sequential_ekf(
            pts=pts,
            point_covs=point_covs,
            center0=center_alg,
            radius0=radius_alg,
            max_sweeps=ekf_max_sweeps,
        )

    if method == "mahal-orthogonal-ls":
        return fit_sphere_mahal_orthogonal_ls(
            pts=pts,
            point_covs=point_covs,
            center0=center_alg,
            radius0=radius_alg,
        )

    if method == "directional-ls":
        return fit_sphere_directional_ls(
            rays=rays,
            measured_distances_mm=measured_distances_mm,
            center0=center_alg,
            radius0=radius_alg,
            pts_for_initial=pts,
            sigma_distance_mm=sigma_distance_mm,
            point_covs=point_covs,
            directional_weight=directional_weight,
        )

    if method == "gauss-newton-geometric":
        return fit_sphere_gauss_newton_geometric(
            pts=pts,
            center0=center_alg,
            radius0=radius_alg,
            max_iter=gn_max_iter,
            damping=gn_damping,
        )

    raise ValueError(f"未知拟合方法：{method}")


# ============================================================
# 输出结果
# ============================================================

def build_method_result(
    method,
    measurements,
    pts,
    rays,
    measured_distances_mm,
    center,
    radius,
    solver_info,
):
    geom_residuals, geom_rms, geom_max = compute_geometric_residuals(
        pts=pts,
        center=center,
        radius=radius,
    )

    dir_residuals, dir_rms, dir_max = compute_directional_residuals(
        rays=rays,
        measured_distances_mm=measured_distances_mm,
        center=center,
        radius=radius,
    )

    point_results = []

    for i, m in enumerate(measurements):
        point_results.append({
            "index": i + 1,
            "point_id": m["point_id"],
            "image_name": m["image_name"],
            "distance_mm": float(m["distance_mm"]),
            "xyz_mm": [
                float(pts[i, 0]),
                float(pts[i, 1]),
                float(pts[i, 2]),
            ],
            "geometric_residual_mm": float(geom_residuals[i]),
            "directional_residual_mm": (
                None if not np.isfinite(dir_residuals[i])
                else float(dir_residuals[i])
            ),
        })

    return {
        "method": method,

        "sphere_center_mm": [
            float(center[0]),
            float(center[1]),
            float(center[2]),
        ],

        "sphere_radius_mm": float(radius),

        "geometric_rms_residual_mm": geom_rms,
        "geometric_max_residual_mm": geom_max,

        "directional_rms_residual_mm": dir_rms,
        "directional_max_residual_mm": dir_max,

        "solver_info": solver_info,
        "point_results": point_results,
    }


def print_xyz_points(measurements, pts):
    print("\n各标定点笛卡尔坐标 (mm)：")
    print(f"{'序号':>4}  {'point_id':>10}  {'image_name':>16}  {'X':>12}  {'Y':>12}  {'Z':>12}")

    for i, m in enumerate(measurements):
        print(
            f"{i + 1:4d}  "
            f"{m['point_id']:>10}  "
            f"{m['image_name']:>16}  "
            f"{pts[i, 0]:12.2f}  "
            f"{pts[i, 1]:12.2f}  "
            f"{pts[i, 2]:12.2f}"
        )


def print_single_method_result(method_result, residual_type="geometric"):
    method = method_result["method"]

    print("\n" + "-" * 80)
    print(f"方法：{method}")
    print("-" * 80)

    cx, cy, cz = method_result["sphere_center_mm"]
    radius = method_result["sphere_radius_mm"]

    print("\n拟合结果：")
    print(f"  球心：({cx:.3f}, {cy:.3f}, {cz:.3f}) mm")
    print(f"  半径：{radius:.3f} mm")

    if residual_type == "directional":
        residual_key = "directional_residual_mm"
        rms_key = "directional_rms_residual_mm"
        max_key = "directional_max_residual_mm"
    else:
        residual_key = "geometric_residual_mm"
        rms_key = "geometric_rms_residual_mm"
        max_key = "geometric_max_residual_mm"

    print("\n各点残差 (mm)：")

    for p in method_result["point_results"]:
        point_id = p["point_id"]
        image_name = p["image_name"]
        residual = p[residual_key]

        if residual is None:
            print(f"  {point_id:>8}  {image_name:>16}      NaN")
        else:
            print(f"  {point_id:>8}  {image_name:>16}  {residual:+.6f}")

    print(f"\nRMS 残差：{method_result[rms_key]:.6f} mm")
    print(f"最大残差：{method_result[max_key]:.6f} mm")


def print_summary_table(method_results):
    print("\n" + "=" * 130)
    print("方法对比汇总")
    print("=" * 130)
    print(
        f"{'method':>28}  "
        f"{'Cx(mm)':>12}  {'Cy(mm)':>12}  {'Cz(mm)':>12}  {'R(mm)':>12}  "
        f"{'Geo RMS':>10}  {'Geo Max':>10}  {'Dir RMS':>10}  {'Dir Max':>10}"
    )

    sorted_items = sorted(
        method_results.items(),
        key=lambda item: item[1]["geometric_rms_residual_mm"],
    )

    for method, r in sorted_items:
        cx, cy, cz = r["sphere_center_mm"]
        radius = r["sphere_radius_mm"]

        print(
            f"{method:>28}  "
            f"{cx:12.3f}  "
            f"{cy:12.3f}  "
            f"{cz:12.3f}  "
            f"{radius:12.3f}  "
            f"{r['geometric_rms_residual_mm']:10.6f}  "
            f"{r['geometric_max_residual_mm']:10.6f}  "
            f"{r['directional_rms_residual_mm']:10.6f}  "
            f"{r['directional_max_residual_mm']:10.6f}"
        )


# ============================================================
# 命令行参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="从全站仪 CSV 点拟合 3D 球面，并对比多种球拟合方法"
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="输入 CSV 文件，例如 input\\single_spot_manifest.csv",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="输出 JSON 文件，例如 output\\sphere_params.json",
    )

    parser.add_argument(
        "--method",
        choices=[
            "algebraic",
            "geometric-ls",
            "geometric-lm",
            "weighted-geometric-ls",
            "weighted-geometric-lm",
            "weighted-marquardt-geometric",
            "weighted-sequential-ekf",
            "mahal-orthogonal-ls",
            "directional-ls",
            "gauss-newton-geometric",
            "all",
        ],
        default="all",
        help="拟合方法。默认 all，会一次性运行全部内置方法。",
    )

    parser.add_argument(
        "--directional-weight",
        choices=["ray-mahalanobis", "distance-only"],
        default="ray-mahalanobis",
        help=(
            "directional-ls 的残差加权：ray-mahalanobis 使用 u^T Σ^{-1} u（与角度/斜距传播协方差一致）；"
            "distance-only 为旧版，仅用斜距 sigma 归一化。"
        ),
    )

    parser.add_argument(
        "--residual-type",
        choices=["geometric", "directional"],
        default="geometric",
        help="每个方法下面打印哪种残差。默认 geometric。",
    )

    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="包含 enabled 不为 1 的点。默认只使用 enabled=1 的点。",
    )

    parser.add_argument(
        "--exclude-point-id",
        nargs="*",
        default=[],
        help="手动排除某些 point_id，例如 --exclude-point-id pt_002 pt_005",
    )

    parser.add_argument(
        "--sigma-angle-arcsec",
        type=float,
        default=5.0,
        help="全站仪水平角（方位）标准差，单位角秒。默认 5.0 arcsec。",
    )

    parser.add_argument(
        "--sigma-v-arcsec",
        type=float,
        default=None,
        help=(
            "竖直角标准差（角秒）。未指定时与 --sigma-angle-arcsec 相同；"
            "分别指定水平/竖直角可更贴近仪器标称协方差。"
        ),
    )

    parser.add_argument(
        "--sigma-distance-mm",
        type=float,
        default=1.0,
        help="全站仪斜距标准差，单位 mm。默认 1.0 mm。",
    )

    parser.add_argument(
        "--gn-max-iter",
        type=int,
        default=100,
        help="Gauss-Newton 最大迭代次数，默认 100。",
    )

    parser.add_argument(
        "--gn-damping",
        type=float,
        default=1e-8,
        help="Gauss-Newton 阻尼系数，默认 1e-8。设为 0 更接近纯 Gauss-Newton。",
    )

    parser.add_argument(
        "--marquardt-max-iter",
        type=int,
        default=200,
        help="weighted-marquardt-geometric 最大外层迭代次数，默认 200。",
    )

    parser.add_argument(
        "--marquardt-lambda-init",
        type=float,
        default=1e-3,
        help="加权 Marquardt 初始阻尼 λ，默认 1e-3。",
    )

    parser.add_argument(
        "--ekf-max-sweeps",
        type=int,
        default=50,
        help=(
            "weighted-sequential-ekf 最大迭代轮数（协方差加权下的迭代高斯–牛顿），默认 50。"
        ),
    )

    return parser.parse_args()


# ============================================================
# 主程序
# ============================================================

def main():
    args = parse_args()

    measurements = load_measurements_from_csv(
        csv_path=args.input,
        enabled_only=not args.include_disabled,
        exclude_point_ids=args.exclude_point_id,
    )

    pts, rays, measured_distances_mm = measurements_to_xyz_and_rays(measurements)

    point_covs = build_point_covariances(
        measurements=measurements,
        sigma_h_arcsec=args.sigma_angle_arcsec,
        sigma_distance_mm=args.sigma_distance_mm,
        sigma_v_arcsec=args.sigma_v_arcsec,
    )

    if args.method == "all":
        methods_to_run = [
            "algebraic",
            "geometric-ls",
            "geometric-lm",
            "weighted-geometric-ls",
            "weighted-geometric-lm",
            "weighted-marquardt-geometric",
            "weighted-sequential-ekf",
            "mahal-orthogonal-ls",
            "directional-ls",
            "gauss-newton-geometric",
        ]
    else:
        methods_to_run = [args.method]

    print("=" * 80)
    print("全站仪 3D 点 → 球面拟合：多方法对比")
    print("=" * 80)
    print(f"输入 CSV：{Path(args.input).resolve()}")
    print(f"有效点数量：{len(measurements)}")
    print(f"水平角标准差：{args.sigma_angle_arcsec} arcsec")
    if args.sigma_v_arcsec is not None:
        print(f"竖直角标准差：{args.sigma_v_arcsec} arcsec")
    else:
        print(f"竖直角标准差：（与水平角相同）{args.sigma_angle_arcsec} arcsec")
    print(f"斜距标准差：{args.sigma_distance_mm} mm")
    print(f"directional-ls 加权：{args.directional_weight}")

    if args.exclude_point_id:
        print(f"排除点：{args.exclude_point_id}")

    print_xyz_points(measurements, pts)

    method_results = {}

    for method in methods_to_run:
        center, radius, solver_info = fit_by_method(
            method=method,
            pts=pts,
            rays=rays,
            measured_distances_mm=measured_distances_mm,
            point_covs=point_covs,
            sigma_distance_mm=args.sigma_distance_mm,
            gn_max_iter=args.gn_max_iter,
            gn_damping=args.gn_damping,
            directional_weight=args.directional_weight,
            marquardt_max_iter=args.marquardt_max_iter,
            marquardt_lambda_init=args.marquardt_lambda_init,
            ekf_max_sweeps=args.ekf_max_sweeps,
        )

        method_result = build_method_result(
            method=method,
            measurements=measurements,
            pts=pts,
            rays=rays,
            measured_distances_mm=measured_distances_mm,
            center=center,
            radius=radius,
            solver_info=solver_info,
        )

        method_results[method] = method_result

        print_single_method_result(
            method_result=method_result,
            residual_type=args.residual_type,
        )

    if len(method_results) > 1:
        print_summary_table(method_results)

    output = {
        "input_csv": str(Path(args.input).resolve()),
        "num_points": len(measurements),
        "coordinate_system": "全站仪坐标系：X=右/东向，Y=前方，Z=向上",
        "sigma_angle_arcsec": float(args.sigma_angle_arcsec),
        "sigma_angle_note": "水平角（方位）标准差；竖直角见 sigma_v_arcsec",
        "sigma_v_arcsec": (
            None if args.sigma_v_arcsec is None else float(args.sigma_v_arcsec)
        ),
        "sigma_distance_mm": float(args.sigma_distance_mm),
        "directional_weight": str(args.directional_weight),
        "gn_max_iter": int(args.gn_max_iter),
        "gn_damping": float(args.gn_damping),
        "marquardt_max_iter": int(args.marquardt_max_iter),
        "marquardt_lambda_init": float(args.marquardt_lambda_init),
        "ekf_max_sweeps": int(args.ekf_max_sweeps),
        "excluded_point_ids": args.exclude_point_id,
        "methods": method_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存至：{output_path.resolve()}")


if __name__ == "__main__":
    main()
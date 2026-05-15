import argparse
import csv
import json
import unicodedata
from pathlib import Path

import numpy as np


SCRIPT_VERSION = "2026-05-15-quadratic6-global-residual"
METHODS = [
    "algebraic",
    "geometric-ls",
    "geometric-lm",
    "weighted-geometric-lm",
]


# ============================================================
# 基础工具
# ============================================================

def dms_to_deg(deg, minutes, sec):
    deg_str = str(deg).strip()
    sign = -1 if deg_str.startswith("-") else 1
    return sign * (
        abs(float(deg_str))
        + float(minutes) / 60.0
        + float(sec) / 3600.0
    )


def measurement_to_unit_vector(az, el):
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

        required = [
            "h_deg", "h_min", "h_sec",
            "v_deg", "v_min", "v_sec",
            "distance_m",
        ]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV 缺少必要列：{missing}")

        for row_index, row in enumerate(reader, start=2):
            point_id = row.get("point_id", f"row_{row_index}")
            enabled = str(row.get("enabled", "1")).strip()

            if enabled_only and enabled not in ("1", "true", "True", "TRUE", "yes", "YES"):
                continue

            if point_id in exclude_point_ids:
                continue

            h_deg = dms_to_deg(row["h_deg"], row["h_min"], row["h_sec"])
            v_deg = dms_to_deg(row["v_deg"], row["v_min"], row["v_sec"])

            measurements.append({
                "point_id": point_id,
                "image_name": row.get("image_name", ""),
                "h_rad": np.radians(h_deg),
                "v_rad": np.radians(v_deg),
                "distance_mm": float(row["distance_m"]) * 1000.0,
            })

    if len(measurements) < 4:
        raise ValueError(
            f"有效点数量不足：{len(measurements)}。拟合 3D 球面至少需要 4 个非共面的点。"
        )

    return measurements


def measurements_to_xyz_and_rays(measurements):
    pts = []
    rays = []
    measured_distances_mm = []

    for m in measurements:
        u = measurement_to_unit_vector(m["h_rad"], m["v_rad"])
        d = m["distance_mm"]

        rays.append(u)
        pts.append(d * u)
        measured_distances_mm.append(d)

    return (
        np.array(pts, dtype=float),
        np.array(rays, dtype=float),
        np.array(measured_distances_mm, dtype=float),
    )


# ============================================================
# 全站仪误差传播：用于 weighted-geometric-lm
# ============================================================

def point_covariance_from_total_station(
    az,
    el,
    d_mm,
    sigma_h_rad,
    sigma_v_rad,
    sigma_distance_mm,
):
    sin_az = np.sin(az)
    cos_az = np.cos(az)
    sin_el = np.sin(el)
    cos_el = np.cos(el)

    dp_daz = d_mm * np.array([
        cos_el * cos_az,
        -cos_el * sin_az,
        0.0,
    ])

    dp_del = d_mm * np.array([
        -sin_el * sin_az,
        -sin_el * cos_az,
        cos_el,
    ])

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

    return J @ sigma_obs @ J.T


def build_point_covariances(
    measurements,
    sigma_angle_arcsec,
    sigma_distance_mm,
    sigma_v_arcsec=None,
):
    sigma_h_rad = np.deg2rad(float(sigma_angle_arcsec) / 3600.0)

    if sigma_v_arcsec is None:
        sigma_v_rad = sigma_h_rad
    else:
        sigma_v_rad = np.deg2rad(float(sigma_v_arcsec) / 3600.0)

    covs = []

    for m in measurements:
        covs.append(
            point_covariance_from_total_station(
                az=m["h_rad"],
                el=m["v_rad"],
                d_mm=m["distance_mm"],
                sigma_h_rad=sigma_h_rad,
                sigma_v_rad=sigma_v_rad,
                sigma_distance_mm=sigma_distance_mm,
            )
        )

    return np.array(covs, dtype=float)


# ============================================================
# 方法 1：algebraic
# ============================================================

def fit_sphere_algebraic(pts):
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
        raise ValueError(f"algebraic 拟合失败，半径平方异常：{radius_sq}")

    center = np.array([cx, cy, cz], dtype=float)
    radius = float(np.sqrt(radius_sq))

    return center, radius, {
        "solver": "numpy.linalg.lstsq",
        "iterations": 1,
        "cost": None,
    }


# ============================================================
# 方法 2：geometric-ls
# ============================================================

def fit_sphere_geometric_ls(pts, center0, radius0):
    try:
        from scipy.optimize import least_squares
    except ImportError as e:
        raise ImportError("需要 scipy。请先运行：pip install scipy") from e

    x0 = np.array([center0[0], center0[1], center0[2], radius0], dtype=float)

    def residual_func(x):
        center = np.array(x[:3], dtype=float)
        radius = x[3]

        if radius <= 0:
            return np.ones(len(pts), dtype=float) * 1e6

        return np.linalg.norm(pts - center, axis=1) - radius

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

    center = np.array(result.x[:3], dtype=float)
    radius = float(result.x[3])

    return center, radius, {
        "solver": "scipy.optimize.least_squares_trf",
        "success": bool(result.success),
        "iterations": int(result.nfev),
        "cost": float(result.cost),
    }


# ============================================================
# 方法 3：geometric-lm
# ============================================================

def pack_log_radius(center, radius):
    return np.array([
        center[0],
        center[1],
        center[2],
        np.log(max(float(radius), 1e-12)),
    ], dtype=float)


def unpack_log_radius(x):
    center = np.array(x[:3], dtype=float)
    radius = float(np.exp(x[3]))
    return center, radius


def fit_sphere_geometric_lm(pts, center0, radius0):
    try:
        from scipy.optimize import least_squares
    except ImportError as e:
        raise ImportError("需要 scipy。请先运行：pip install scipy") from e

    x0 = pack_log_radius(center0, radius0)

    def residual_func(x):
        center, radius = unpack_log_radius(x)
        return np.linalg.norm(pts - center, axis=1) - radius

    def jac_func(x):
        center, radius = unpack_log_radius(x)
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

    center, radius = unpack_log_radius(result.x)

    return center, radius, {
        "solver": "scipy.optimize.least_squares_lm_log_radius",
        "success": bool(result.success),
        "iterations": int(result.nfev),
        "cost": float(result.cost),
    }


# ============================================================
# 方法 4：weighted-geometric-lm
# ============================================================

def fit_sphere_weighted_geometric_lm(pts, point_covs, center0, radius0):
    try:
        from scipy.optimize import least_squares
    except ImportError as e:
        raise ImportError("需要 scipy。请先运行：pip install scipy") from e

    x0 = pack_log_radius(center0, radius0)

    def residual_func(x):
        center, radius = unpack_log_radius(x)

        vecs = pts - center
        dists = np.linalg.norm(vecs, axis=1)
        dists = np.maximum(dists, 1e-12)

        normals = vecs / dists[:, None]
        geom_residuals = dists - radius

        residuals = []

        for i in range(len(pts)):
            n = normals[i]
            cov = point_covs[i]

            var_e = float(n.T @ cov @ n)
            var_e = max(var_e, 1e-12)

            residuals.append(geom_residuals[i] / np.sqrt(var_e))

        return np.array(residuals, dtype=float)

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

    center, radius = unpack_log_radius(result.x)

    return center, radius, {
        "solver": "scipy.optimize.least_squares_weighted_lm_log_radius",
        "success": bool(result.success),
        "iterations": int(result.nfev),
        "cost": float(result.cost),
    }


# ============================================================
# 残差计算
# ============================================================

def compute_geometric_residuals(pts, center, radius):
    residuals = np.linalg.norm(pts - center, axis=1) - radius
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    max_abs = float(np.max(np.abs(residuals)))
    return residuals, rms, max_abs


def ray_sphere_intersection_distance(ray_u, center, radius, measured_distance_mm):
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


def compute_directional_residuals(rays, measured_distances_mm, center, radius):
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
# 拟合入口
# ============================================================

def fit_by_method(method, pts, point_covs, center0, radius0, algebraic_info):
    if method == "algebraic":
        return center0, radius0, algebraic_info

    if method == "geometric-ls":
        return fit_sphere_geometric_ls(pts, center0, radius0)

    if method == "geometric-lm":
        return fit_sphere_geometric_lm(pts, center0, radius0)

    if method == "weighted-geometric-lm":
        return fit_sphere_weighted_geometric_lm(pts, point_covs, center0, radius0)

    raise ValueError(f"未知方法：{method}")


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
        pts, center, radius
    )

    dir_residuals, dir_rms, dir_max = compute_directional_residuals(
        rays, measured_distances_mm, center, radius
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
        "sphere_equation_mm": (
            f"(X - {center[0]:.6f})^2 + "
            f"(Y - {center[1]:.6f})^2 + "
            f"(Z - {center[2]:.6f})^2 = {radius:.6f}^2"
        ),
        "geometric_rms_residual_mm": geom_rms,
        "geometric_max_residual_mm": geom_max,
        "directional_rms_residual_mm": dir_rms,
        "directional_max_residual_mm": dir_max,
        "solver_info": solver_info,
        "point_results": point_results,
    }


# ============================================================
# 打印输出
# ============================================================

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
    print(f"  球面方法：{method}")
    print(f"  球心：({cx:.3f}, {cy:.3f}, {cz:.3f}) mm")
    print(f"  半径：{radius:.3f} mm")
    print(f"  球面方程：{method_result['sphere_equation_mm']}")

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


def select_best_method(method_results):
    return min(
        method_results.items(),
        key=lambda item: item[1]["geometric_rms_residual_mm"],
    )


def print_best_method(best_method, best_result):
    cx, cy, cz = best_result["sphere_center_mm"]
    radius = best_result["sphere_radius_mm"]

    print("\n" + "=" * 80)
    print("精度最高方法")
    print("=" * 80)
    print("判定标准：几何 RMS 残差最小")
    print(f"球面方法：{best_method}")
    print(f"球心：({cx:.6f}, {cy:.6f}, {cz:.6f}) mm")
    print(f"半径：{radius:.6f} mm")
    print(f"球面方程：{best_result['sphere_equation_mm']}")
    print(f"几何 RMS 残差：{best_result['geometric_rms_residual_mm']:.6f} mm")
    print(f"几何最大残差：{best_result['geometric_max_residual_mm']:.6f} mm")
    print(f"方向 RMS 残差：{best_result['directional_rms_residual_mm']:.6f} mm")
    print(f"方向最大残差：{best_result['directional_max_residual_mm']:.6f} mm")



# ============================================================
# 自由曲面模型：基准球 + 全局低阶残差 + 可选局部 RBF 修正
# ============================================================

# 说明：
# 1) 基准球仍然使用上面四种方法中几何 RMS 最小的最佳球。
# 2) 径向残差定义为：||P - C|| - R。
#    residual > 0 表示该点相对理想球面向外鼓出；
#    residual < 0 表示该点相对理想球面向内凹进。
# 3) 全局低阶残差严格使用文档公式：q(u)=a1*ux^2+a2*uy^2+a3*uz^2+a4*ux*uy+a5*ux*uz+a6*uy*uz。
# 4) 局部修正使用球面弧长距离 + Wendland C2 紧支撑核。


def compute_unit_directions_and_radial_residuals(pts, center, radius):
    vecs = pts - center
    norms = np.linalg.norm(vecs, axis=1)
    norms = np.maximum(norms, 1e-12)
    unit_dirs = vecs / norms[:, None]
    radial_residuals = norms - radius
    return unit_dirs, radial_residuals


def generate_global_quadratic_powers():
    """
    文档公式的第一层全局低阶残差：

        q(u) = a1*ux^2 + a2*uy^2 + a3*uz^2
             + a4*ux*uy + a5*ux*uz + a6*uy*uz

    注意：这里不是完整二阶多项式。
    不包含常数项 1，也不包含一次项 ux、uy、uz。
    """
    return [
        (2, 0, 0),  # a1 * ux^2
        (0, 2, 0),  # a2 * uy^2
        (0, 0, 2),  # a3 * uz^2
        (1, 1, 0),  # a4 * ux * uy
        (1, 0, 1),  # a5 * ux * uz
        (0, 1, 1),  # a6 * uy * uz
    ]


def polynomial_power_label(power):
    label_map = {
        (2, 0, 0): "u_x^2",
        (0, 2, 0): "u_y^2",
        (0, 0, 2): "u_z^2",
        (1, 1, 0): "u_x*u_y",
        (1, 0, 1): "u_x*u_z",
        (0, 1, 1): "u_y*u_z",
    }
    if tuple(power) in label_map:
        return label_map[tuple(power)]

    names = ["u_x", "u_y", "u_z"]
    parts = []
    for name, pwr in zip(names, power):
        if pwr == 0:
            continue
        if pwr == 1:
            parts.append(name)
        else:
            parts.append(f"{name}^{pwr}")
    return "1" if not parts else "*".join(parts)


def build_global_design_matrix(unit_dirs, powers):
    unit_dirs = np.asarray(unit_dirs, dtype=float)
    A = np.ones((len(unit_dirs), len(powers)), dtype=float)

    for j, (px, py, pz) in enumerate(powers):
        if px:
            A[:, j] *= unit_dirs[:, 0] ** px
        if py:
            A[:, j] *= unit_dirs[:, 1] ** py
        if pz:
            A[:, j] *= unit_dirs[:, 2] ** pz

    return A


def solve_regularized_least_squares(A, y, ridge_lambda=0.0, regularize_first=False):
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    ridge_lambda = float(ridge_lambda)

    if ridge_lambda > 0:
        reg = np.eye(A.shape[1], dtype=float)
        if not regularize_first:
            reg[0, 0] = 0.0  # 全局多项式的常数项不做正则，避免整体偏差被压低
        A_aug = np.vstack([A, np.sqrt(ridge_lambda) * reg])
        y_aug = np.concatenate([y, np.zeros(A.shape[1], dtype=float)])
        coeffs, _, rank, singular_values = np.linalg.lstsq(A_aug, y_aug, rcond=None)
    else:
        coeffs, _, rank, singular_values = np.linalg.lstsq(A, y, rcond=None)

    return coeffs, int(rank), singular_values


def fit_global_low_order_residual_model(
    unit_dirs,
    radial_residuals,
    degree=2,
    ridge_lambda=1e-8,
):
    # 按用户文档中的公式固定使用 6 个二次项。
    # 传入的 degree 参数仅保留为命令行兼容，不再决定全局残差项数。
    requested_degree = int(degree)
    powers = generate_global_quadratic_powers()
    A = build_global_design_matrix(unit_dirs, powers)
    coeffs, rank, singular_values = solve_regularized_least_squares(
        A=A,
        y=radial_residuals,
        ridge_lambda=ridge_lambda,
        regularize_first=True,
    )

    fitted = A @ coeffs
    errors = radial_residuals - fitted

    if len(singular_values) > 0 and float(np.min(singular_values)) > 0:
        condition_number = float(np.max(singular_values) / np.min(singular_values))
    else:
        condition_number = None

    return {
        "model_type": "paper_quadratic_global_residual_q_u",
        "formula": "q(u)=a1*u_x^2+a2*u_y^2+a3*u_z^2+a4*u_x*u_y+a5*u_x*u_z+a6*u_y*u_z",
        "degree": 2,
        "requested_degree_argument": requested_degree,
        "ridge_lambda": float(ridge_lambda),
        "num_terms": len(powers),
        "num_unknowns": len(powers),
        "powers": [list(p) for p in powers],
        "term_labels": [polynomial_power_label(p) for p in powers],
        "coefficient_names": ["a1", "a2", "a3", "a4", "a5", "a6"],
        "coefficients_mm": [float(c) for c in coeffs],
        "rank": rank,
        "singular_values": [float(s) for s in singular_values],
        "condition_number": condition_number,
        "input_residual_stats": residual_stats(radial_residuals),
        "fit_stats": residual_stats(errors),
        "fitted_residuals_mm": [float(v) for v in fitted],
        "reliability_notes": global_model_reliability_notes(
            num_points=len(radial_residuals),
            num_terms=len(powers),
            degree=degree,
        ),
    }


def predict_global_low_order_residual(unit_dirs, model):
    powers = [tuple(p) for p in model["powers"]]
    coeffs = np.asarray(model["coefficients_mm"], dtype=float)
    A = build_global_design_matrix(unit_dirs, powers)
    return A @ coeffs


def residual_stats(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)

    if not np.any(finite):
        return {
            "count": 0,
            "mean_mm": None,
            "mean_abs_mm": None,
            "rms_mm": None,
            "max_abs_mm": None,
            "std_mm": None,
        }

    v = values[finite]
    return {
        "count": int(len(v)),
        "mean_mm": float(np.mean(v)),
        "mean_abs_mm": float(np.mean(np.abs(v))),
        "rms_mm": float(np.sqrt(np.mean(v ** 2))),
        "max_abs_mm": float(np.max(np.abs(v))),
        "std_mm": float(np.std(v)),
    }


def leave_one_out_global_residual_validation(
    measurements,
    unit_dirs,
    radial_residuals,
    degree=2,
    ridge_lambda=1e-8,
):
    point_validations = []
    prediction_errors = []

    n = len(radial_residuals)
    if n < 3:
        raise ValueError("全局低阶残差留一验证至少需要 3 个点")

    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False

        model_i = fit_global_low_order_residual_model(
            unit_dirs=unit_dirs[train_mask],
            radial_residuals=radial_residuals[train_mask],
            degree=degree,
            ridge_lambda=ridge_lambda,
        )

        predicted = float(predict_global_low_order_residual(unit_dirs[i:i + 1], model_i)[0])
        actual = float(radial_residuals[i])
        error = actual - predicted
        prediction_errors.append(error)

        m = measurements[i]
        point_validations.append({
            "index": i + 1,
            "point_id": m["point_id"],
            "image_name": m["image_name"],
            "actual_radial_residual_mm": actual,
            "predicted_global_residual_mm": predicted,
            "validation_error_mm": float(error),
            "abs_validation_error_mm": float(abs(error)),
        })

    return {
        "validation_type": "leave_one_out_one_point_each_time",
        "description": "每次取 1 个点做验证，其余点拟合全局低阶残差曲面；最后统计所有取点验证误差。",
        "model_formula": "q(u)=a1*u_x^2+a2*u_y^2+a3*u_z^2+a4*u_x*u_y+a5*u_x*u_z+a6*u_y*u_z",
        "degree": 2,
        "requested_degree_argument": int(degree),
        "ridge_lambda": float(ridge_lambda),
        "point_validations": point_validations,
        "summary": residual_stats(prediction_errors),
    }



def display_width(text):
    """按终端显示宽度估算字符串宽度，中文字符按 2 列计算。"""
    text = str(text)
    width = 0
    for ch in text:
        if unicodedata.east_asian_width(ch) in ("F", "W"):
            width += 2
        else:
            width += 1
    return width


def format_cell(value, width, align="right", precision=None):
    if isinstance(value, float) and precision is not None:
        text = f"{value:.{precision}f}"
    else:
        text = str(value)

    pad = max(int(width) - display_width(text), 0)
    if align == "left":
        return text + " " * pad
    return " " * pad + text


def print_table(headers, rows, aligns=None, precisions=None, min_gap=2):
    headers = list(headers)
    rows = list(rows)
    aligns = aligns or ["right"] * len(headers)
    precisions = precisions or [None] * len(headers)

    widths = [display_width(h) for h in headers]
    formatted_rows = []

    for row in rows:
        formatted_row = []
        for i, value in enumerate(row):
            if isinstance(value, float) and precisions[i] is not None:
                text = f"{value:.{precisions[i]}f}"
            else:
                text = str(value)
            formatted_row.append(text)
            widths[i] = max(widths[i], display_width(text))
        formatted_rows.append(formatted_row)

    sep = " " * int(min_gap)
    print(sep.join(format_cell(h, widths[i], aligns[i]) for i, h in enumerate(headers)))
    for row in formatted_rows:
        print(sep.join(format_cell(v, widths[i], aligns[i]) for i, v in enumerate(row)))


def global_model_reliability_notes(num_points, num_terms, degree):
    notes = []
    loo_train_points = max(int(num_points) - 1, 0)

    notes.append(
        "全局低阶残差已按文档公式固定为 6 个二次项：ux^2、uy^2、uz^2、uxuy、uxuz、uyuz。"
    )

    if int(num_points) <= int(num_terms):
        notes.append(
            f"全点拟合为欠约束或刚约束：点数={num_points}，模型项数={num_terms}。"
        )
    elif int(num_points) < 3 * int(num_terms):
        notes.append(
            f"全点拟合约束偏弱：点数={num_points}，模型项数={num_terms}，建议至少达到约 3 倍以上。"
        )

    if loo_train_points <= int(num_terms):
        notes.append(
            f"留一法训练集欠约束或刚约束：每次训练点数={loo_train_points}，模型项数={num_terms}，预测会非常不稳定。"
        )
    elif loo_train_points < 3 * int(num_terms):
        notes.append(
            f"留一法训练集约束很弱：每次训练点数={loo_train_points}，模型项数={num_terms}，预测误差可能被放大。"
        )

    return notes

def print_global_residual_model_summary(global_model, loo_result):
    print("\n" + "=" * 80)
    print("自由曲面第一层：全局低阶残差模型")
    print("=" * 80)
    print("模型：文档公式二次型全局残差 q(u)")
    print(f"公式：{global_model.get('formula')}")
    print(f"未知数数量：{global_model['num_unknowns']} 个（a1~a6）")
    if global_model.get('requested_degree_argument') != 2:
        print(f"说明：命令行 --global-degree={global_model.get('requested_degree_argument')} 已保留兼容，但当前模型固定使用文档公式的 6 个二次项。")
    print(f"岭正则 lambda：{global_model['ridge_lambda']}")
    print(f"设计矩阵 rank：{global_model['rank']} / {global_model['num_terms']}")

    coeff_rows = []
    for name, term, value in zip(
        global_model.get('coefficient_names', []),
        global_model.get('term_labels', []),
        global_model.get('coefficients_mm', []),
    ):
        coeff_rows.append([name, term, value])
    if coeff_rows:
        print("\n全局低阶残差系数 (mm)：")
        print_table(
            headers=["系数", "基函数", "值"],
            rows=coeff_rows,
            aligns=["right", "right", "right"],
            precisions=[None, None, 6],
        )

    if global_model.get("condition_number") is None:
        print("设计矩阵条件数：无法计算，可能存在奇异或近奇异问题")
    else:
        print(f"设计矩阵条件数：{global_model['condition_number']:.6e}")

    print("径向残差定义：||P-C||-R；正值=鼓出，负值=凹进")

    notes = global_model.get("reliability_notes") or []
    loo_notes = global_model_reliability_notes(
        num_points=loo_result['summary']['count'],
        num_terms=global_model['num_terms'],
        degree=global_model['degree'],
    )
    notes = list(dict.fromkeys(notes + loo_notes))
    if notes:
        print("\n重要诊断：")
        for note in notes:
            print(f"  - {note}")

    input_stats = global_model.get("input_residual_stats", {})
    fit_stats = global_model["fit_stats"]

    print("\n全局模型修正前后对比，使用全部点拟合 (mm)：")
    print_table(
        headers=["阶段", "RMS", "平均绝对值", "最大绝对值"],
        rows=[
            ["最佳球原始径向残差", input_stats.get('rms_mm'), input_stats.get('mean_abs_mm'), input_stats.get('max_abs_mm')],
            ["全局低阶修正后剩余误差", fit_stats.get('rms_mm'), fit_stats.get('mean_abs_mm'), fit_stats.get('max_abs_mm')],
        ],
        aligns=["left", "right", "right", "right"],
        precisions=[None, 6, 6, 6],
    )

    print("\n留一法验证说明：")
    print("  预测残差 = 当前点不参与拟合时，由其余点拟合出的全局残差模型对当前点的预测")
    print("  验证误差 = 实测残差 - 预测残差")
    print("  绝对误差 = |验证误差|")

    rows = []
    for item in loo_result["point_validations"]:
        rows.append([
            item['point_id'],
            item['image_name'],
            item['actual_radial_residual_mm'],
            item['predicted_global_residual_mm'],
            item['validation_error_mm'],
            item['abs_validation_error_mm'],
        ])

    print("\n留一法验证：每次取 1 点验证，其余点拟合全局残差曲面")
    print_table(
        headers=["point_id", "image_name", "实测残差", "预测残差", "验证误差", "绝对误差"],
        rows=rows,
        aligns=["right", "right", "right", "right", "right", "right"],
        precisions=[None, None, 6, 6, 6, 6],
    )

    summary = loo_result["summary"]
    print("\n留一法验证平均结果 (mm)：")
    print(f"  平均误差 bias：{summary['mean_mm']:.6f}  （正负误差会抵消，只看系统偏向，不代表精度）")
    print(f"  平均绝对误差 MAE：{summary['mean_abs_mm']:.6f}  （更能代表平均预测误差）")
    print(f"  RMS：{summary['rms_mm']:.6f}  （对大误差更敏感）")
    print(f"  最大绝对误差：{summary['max_abs_mm']:.6f}")

def angular_distance_on_unit_sphere(unit_dirs_a, unit_dirs_b):
    unit_dirs_a = np.asarray(unit_dirs_a, dtype=float)
    unit_dirs_b = np.asarray(unit_dirs_b, dtype=float)
    cos_values = unit_dirs_a @ unit_dirs_b.T
    cos_values = np.clip(cos_values, -1.0, 1.0)
    return np.arccos(cos_values)


def sphere_arc_length_distance(unit_dirs_a, unit_dirs_b, radius):
    return float(radius) * angular_distance_on_unit_sphere(unit_dirs_a, unit_dirs_b)


def wendland_c2_kernel(q):
    q = np.asarray(q, dtype=float)
    values = np.zeros_like(q, dtype=float)
    mask = (q >= 0.0) & (q < 1.0)
    qm = q[mask]
    values[mask] = (1.0 - qm) ** 4 * (4.0 * qm + 1.0)
    return values


def estimate_default_rbf_radius_mm(unit_dirs, base_radius):
    n = len(unit_dirs)
    if n < 2:
        return float(base_radius) * 0.25

    arc = sphere_arc_length_distance(unit_dirs, unit_dirs, base_radius)
    arc = np.asarray(arc, dtype=float)
    arc[arc <= 1e-12] = np.nan
    nearest = np.nanmin(arc, axis=1)
    nearest = nearest[np.isfinite(nearest)]

    if len(nearest) == 0:
        return float(base_radius) * 0.25

    # 半径取最近邻弧长中位数的 2.5 倍，使局部核既有重叠又保持局部性。
    return float(np.median(nearest) * 2.5)


def fit_local_rbf_residual_model(
    unit_dirs,
    target_local_residuals,
    base_radius,
    support_radius_mm=None,
    ridge_lambda=1e-8,
):
    unit_dirs = np.asarray(unit_dirs, dtype=float)
    target_local_residuals = np.asarray(target_local_residuals, dtype=float)

    if support_radius_mm is None or float(support_radius_mm) <= 0:
        support_radius_mm = estimate_default_rbf_radius_mm(unit_dirs, base_radius)

    support_radius_mm = float(support_radius_mm)
    arc = sphere_arc_length_distance(unit_dirs, unit_dirs, base_radius)
    K = wendland_c2_kernel(arc / support_radius_mm)

    coeffs, rank, singular_values = solve_regularized_least_squares(
        A=K,
        y=target_local_residuals,
        ridge_lambda=ridge_lambda,
        regularize_first=True,
    )

    fitted = K @ coeffs
    errors = target_local_residuals - fitted

    return {
        "model_type": "local_rbf_wendland_c2_on_spherical_arc_length",
        "kernel": "Wendland C2: phi(q)=(1-q)^4*(4q+1), 0<=q<1, otherwise 0",
        "distance": "spherical_arc_length_mm = base_radius_mm * arccos(dot(u_i, u_j))",
        "support_radius_mm": support_radius_mm,
        "ridge_lambda": float(ridge_lambda),
        "num_centers": int(len(unit_dirs)),
        "centers_unit_dirs": unit_dirs.tolist(),
        "weights_mm": [float(v) for v in coeffs],
        "rank": rank,
        "singular_values": [float(s) for s in singular_values],
        "target_local_residuals_mm": [float(v) for v in target_local_residuals],
        "fitted_local_residuals_mm": [float(v) for v in fitted],
        "fit_stats": residual_stats(errors),
    }


def predict_local_rbf_residual(unit_dirs, rbf_model, base_radius):
    unit_dirs = np.asarray(unit_dirs, dtype=float)
    centers = np.asarray(rbf_model["centers_unit_dirs"], dtype=float)
    weights = np.asarray(rbf_model["weights_mm"], dtype=float)
    support_radius_mm = float(rbf_model["support_radius_mm"])

    arc = sphere_arc_length_distance(unit_dirs, centers, base_radius)
    K = wendland_c2_kernel(arc / support_radius_mm)
    return K @ weights


def build_surface_point_results(
    measurements,
    pts,
    center,
    radius,
    global_model,
    rbf_model=None,
):
    unit_dirs, actual_radial_residuals = compute_unit_directions_and_radial_residuals(
        pts=pts,
        center=center,
        radius=radius,
    )

    global_pred = predict_global_low_order_residual(unit_dirs, global_model)

    if rbf_model is None:
        local_pred = np.zeros(len(pts), dtype=float)
    else:
        local_pred = predict_local_rbf_residual(unit_dirs, rbf_model, radius)

    total_pred = global_pred + local_pred
    corrected_errors = actual_radial_residuals - total_pred

    rows = []
    for i, m in enumerate(measurements):
        rows.append({
            "index": i + 1,
            "point_id": m["point_id"],
            "image_name": m["image_name"],
            "actual_radial_residual_mm": float(actual_radial_residuals[i]),
            "predicted_global_residual_mm": float(global_pred[i]),
            "predicted_local_rbf_residual_mm": float(local_pred[i]),
            "predicted_total_residual_mm": float(total_pred[i]),
            "corrected_surface_error_mm": float(corrected_errors[i]),
            "corrected_radius_mm": float(radius + total_pred[i]),
        })

    return {
        "point_results": rows,
        "summary": residual_stats(corrected_errors),
    }


def print_rbf_model_summary(rbf_model, rbf_point_results):
    print("\n" + "=" * 80)
    print("自由曲面第二层：局部 RBF 修正")
    print("=" * 80)
    print(f"RBF 中心数量：{rbf_model['num_centers']}")
    print(f"影响半径 rho：{rbf_model['support_radius_mm']:.6f} mm")
    print(f"岭正则 lambda：{rbf_model['ridge_lambda']}")
    print("核函数：Wendland C2 紧支撑核")

    stats = rbf_model["fit_stats"]
    print("\n局部 RBF 对新增点局部残差的拟合剩余误差 (mm)：")
    print(f"  RMS：{stats['rms_mm']:.6f}")
    print(f"  平均绝对值：{stats['mean_abs_mm']:.6f}")
    print(f"  最大绝对值：{stats['max_abs_mm']:.6f}")

    corrected_stats = rbf_point_results["summary"]
    print("\n新增点经过 全局残差 + 局部 RBF 修正 后的剩余误差 (mm)：")
    print(f"  RMS：{corrected_stats['rms_mm']:.6f}")
    print(f"  平均绝对值：{corrected_stats['mean_abs_mm']:.6f}")
    print(f"  最大绝对值：{corrected_stats['max_abs_mm']:.6f}")


def build_freeform_surface_model(
    base_measurements,
    base_pts,
    center,
    radius,
    global_degree=1,
    global_ridge_lambda=1e-8,
    rbf_measurements=None,
    rbf_pts=None,
    rbf_support_radius_mm=None,
    rbf_ridge_lambda=1e-8,
):
    base_unit_dirs, base_radial_residuals = compute_unit_directions_and_radial_residuals(
        pts=base_pts,
        center=center,
        radius=radius,
    )

    global_model = fit_global_low_order_residual_model(
        unit_dirs=base_unit_dirs,
        radial_residuals=base_radial_residuals,
        degree=global_degree,
        ridge_lambda=global_ridge_lambda,
    )

    global_loo = leave_one_out_global_residual_validation(
        measurements=base_measurements,
        unit_dirs=base_unit_dirs,
        radial_residuals=base_radial_residuals,
        degree=global_degree,
        ridge_lambda=global_ridge_lambda,
    )

    print_global_residual_model_summary(global_model, global_loo)

    base_surface_results = build_surface_point_results(
        measurements=base_measurements,
        pts=base_pts,
        center=center,
        radius=radius,
        global_model=global_model,
        rbf_model=None,
    )

    rbf_model = None
    rbf_surface_results = None
    rbf_input_summary = None

    if rbf_measurements is not None and rbf_pts is not None:
        rbf_unit_dirs, rbf_radial_residuals = compute_unit_directions_and_radial_residuals(
            pts=rbf_pts,
            center=center,
            radius=radius,
        )

        rbf_global_pred = predict_global_low_order_residual(rbf_unit_dirs, global_model)
        target_local_residuals = rbf_radial_residuals - rbf_global_pred

        rbf_model = fit_local_rbf_residual_model(
            unit_dirs=rbf_unit_dirs,
            target_local_residuals=target_local_residuals,
            base_radius=radius,
            support_radius_mm=rbf_support_radius_mm,
            ridge_lambda=rbf_ridge_lambda,
        )

        rbf_surface_results = build_surface_point_results(
            measurements=rbf_measurements,
            pts=rbf_pts,
            center=center,
            radius=radius,
            global_model=global_model,
            rbf_model=rbf_model,
        )

        rbf_input_summary = {
            "num_points": int(len(rbf_measurements)),
            "radial_residual_stats_before_correction": residual_stats(rbf_radial_residuals),
            "global_only_error_stats_on_rbf_points": residual_stats(target_local_residuals),
        }

        print_rbf_model_summary(rbf_model, rbf_surface_results)

    return {
        "model_name": "base_sphere_plus_global_low_order_residual_plus_optional_local_rbf",
        "base_sphere_center_mm": [float(center[0]), float(center[1]), float(center[2])],
        "base_sphere_radius_mm": float(radius),
        "radial_residual_definition": "||P-C||-R；正值表示鼓出，负值表示凹进",
        "global_low_order_residual_model": global_model,
        "global_leave_one_out_validation": global_loo,
        "base_points_after_global_correction": base_surface_results,
        "local_rbf_model": rbf_model,
        "rbf_input_summary": rbf_input_summary,
        "rbf_points_after_global_plus_rbf_correction": rbf_surface_results,
    }

# ============================================================
# 命令行参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="从全站仪 CSV 点拟合 3D 球面，并输出最佳方法"
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
        choices=METHODS + ["all"],
        default="all",
        help="拟合方法。默认 all，运行全部保留方法。",
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
        help="全站仪水平角标准差，单位角秒。默认 5.0 arcsec。",
    )

    parser.add_argument(
        "--sigma-v-arcsec",
        type=float,
        default=None,
        help="竖直角标准差，单位角秒。未指定时与水平角相同。",
    )

    parser.add_argument(
        "--sigma-distance-mm",
        type=float,
        default=1.0,
        help="全站仪斜距标准差，单位 mm。默认 1.0 mm。",
    )

    parser.add_argument(
        "--global-degree",
        type=int,
        default=2,
        help="保留兼容参数。当前全局低阶残差固定使用文档公式 q(u) 的 6 个二次项，默认 2，此参数不再改变项数。",
    )

    parser.add_argument(
        "--global-ridge-lambda",
        type=float,
        default=1e-8,
        help="全局低阶残差拟合的岭正则系数。默认 1e-8。",
    )

    parser.add_argument(
        "--rbf-input",
        default=None,
        help="可选：用于局部 RBF 修正的新增全站仪 CSV 文件。不提供时只做最佳球 + 全局低阶残差。",
    )

    parser.add_argument(
        "--rbf-include-disabled",
        action="store_true",
        help="RBF 输入文件中包含 enabled 不为 1 的点。默认只使用 enabled=1 的点。",
    )

    parser.add_argument(
        "--rbf-exclude-point-id",
        nargs="*",
        default=[],
        help="RBF 输入文件中手动排除某些 point_id，例如 --rbf-exclude-point-id pt_002 pt_005",
    )

    parser.add_argument(
        "--rbf-support-radius-mm",
        type=float,
        default=None,
        help="局部 RBF 影响半径 rho，单位 mm。不指定时根据新增点最近邻球面弧长自动估计。",
    )

    parser.add_argument(
        "--rbf-ridge-lambda",
        type=float,
        default=1e-8,
        help="局部 RBF 权重求解的岭正则系数。默认 1e-8。",
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
        sigma_angle_arcsec=args.sigma_angle_arcsec,
        sigma_distance_mm=args.sigma_distance_mm,
        sigma_v_arcsec=args.sigma_v_arcsec,
    )

    if args.method == "all":
        methods_to_run = list(METHODS)
    else:
        methods_to_run = [args.method]

    print("=" * 80)
    print("全站仪 3D 点 → 球面拟合：多方法对比")
    print(f"脚本版本：{SCRIPT_VERSION}")
    print("=" * 80)
    print(f"输入 CSV：{Path(args.input).resolve()}")
    print(f"有效点数量：{len(measurements)}")
    print(f"水平角标准差：{args.sigma_angle_arcsec} arcsec")

    if args.sigma_v_arcsec is not None:
        print(f"竖直角标准差：{args.sigma_v_arcsec} arcsec")
    else:
        print(f"竖直角标准差：（与水平角相同）{args.sigma_angle_arcsec} arcsec")

    print(f"斜距标准差：{args.sigma_distance_mm} mm")
    print(f"输出方法：{', '.join(methods_to_run)}")

    if args.exclude_point_id:
        print(f"排除点：{args.exclude_point_id}")

    print_xyz_points(measurements, pts)

    center0, radius0, algebraic_info = fit_sphere_algebraic(pts)

    method_results = {}

    for method in methods_to_run:
        center, radius, solver_info = fit_by_method(
            method=method,
            pts=pts,
            point_covs=point_covs,
            center0=center0,
            radius0=radius0,
            algebraic_info=algebraic_info,
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

    best_method, best_result = select_best_method(method_results)
    print_best_method(best_method, best_result)

    best_center = np.array(best_result["sphere_center_mm"], dtype=float)
    best_radius = float(best_result["sphere_radius_mm"])

    rbf_measurements = None
    rbf_pts = None
    rbf_input_csv = None

    if args.rbf_input:
        rbf_input_csv = str(Path(args.rbf_input).resolve())
        print("\n" + "=" * 80)
        print("读取局部 RBF 修正点")
        print("=" * 80)
        print(f"RBF 输入 CSV：{rbf_input_csv}")

        rbf_measurements = load_measurements_from_csv(
            csv_path=args.rbf_input,
            enabled_only=not args.rbf_include_disabled,
            exclude_point_ids=args.rbf_exclude_point_id,
        )
        rbf_pts, _, _ = measurements_to_xyz_and_rays(rbf_measurements)
        print(f"RBF 有效点数量：{len(rbf_measurements)}")

        if args.rbf_exclude_point_id:
            print(f"RBF 排除点：{args.rbf_exclude_point_id}")
    else:
        print("\n未提供 --rbf-input：本次只执行 最佳拟合球 + 全局低阶残差 优化。")

    freeform_surface_model = build_freeform_surface_model(
        base_measurements=measurements,
        base_pts=pts,
        center=best_center,
        radius=best_radius,
        global_degree=args.global_degree,
        global_ridge_lambda=args.global_ridge_lambda,
        rbf_measurements=rbf_measurements,
        rbf_pts=rbf_pts,
        rbf_support_radius_mm=args.rbf_support_radius_mm,
        rbf_ridge_lambda=args.rbf_ridge_lambda,
    )

    output = {
        "input_csv": str(Path(args.input).resolve()),
        "rbf_input_csv": rbf_input_csv,
        "num_points": len(measurements),
        "coordinate_system": "全站仪坐标系：X=右/东向，Y=前方，Z=向上",
        "sigma_angle_arcsec": float(args.sigma_angle_arcsec),
        "sigma_v_arcsec": (
            None if args.sigma_v_arcsec is None else float(args.sigma_v_arcsec)
        ),
        "sigma_distance_mm": float(args.sigma_distance_mm),
        "methods": method_results,
        "best_method_by_geometric_rms": best_method,
        "best_result": best_result,
        "freeform_surface_model": freeform_surface_model,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存至：{output_path.resolve()}")


if __name__ == "__main__":
    main()
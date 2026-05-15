import argparse
import csv
import json
from pathlib import Path

import numpy as np


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

    output = {
        "input_csv": str(Path(args.input).resolve()),
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
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存至：{output_path.resolve()}")


if __name__ == "__main__":
    main()
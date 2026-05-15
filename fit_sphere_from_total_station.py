
import os
import csv
import json
import argparse
from pathlib import Path

import numpy as np


def dms_to_deg(deg, minutes, sec):
    """
    度分秒 → 十进制度

    注意：
    这里 deg 允许传字符串，是为了保留 "-0" 这种情况。
    比如 "-0, 51, 40" 应该表示 -0°51′40″。
    """
    deg_str = str(deg).strip()

    sign = -1 if deg_str.startswith("-") else 1

    deg_abs = abs(float(deg_str))
    minutes = float(minutes)
    sec = float(sec)

    return sign * (deg_abs + minutes / 60.0 + sec / 3600.0)


def load_measurements_from_csv(csv_path, enabled_only=True):
    """
    从 CSV 读取测量点。

    必需字段：
      h_deg,h_min,h_sec
      v_deg,v_min,v_sec
      distance_m

    可选字段：
      enabled
      point_id
      image_name
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到输入 CSV 文件：{csv_path}")

    measurements = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required_columns = [
            "h_deg", "h_min", "h_sec",
            "v_deg", "v_min", "v_sec",
            "distance_m",
        ]

        missing_columns = [c for c in required_columns if c not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"CSV 缺少必要列：{missing_columns}")

        for row_index, row in enumerate(reader, start=2):
            try:
                enabled_value = str(row.get("enabled", "1")).strip()

                if enabled_only and enabled_value not in ("1", "true", "True", "TRUE", "yes", "YES"):
                    continue

                distance_m = float(row["distance_m"])
                distance_mm = distance_m * 1000.0

                measurement = {
                    "image_name": row.get("image_name", ""),
                    "point_id": row.get("point_id", f"row_{row_index}"),

                    "h_deg": row["h_deg"],
                    "h_min": row["h_min"],
                    "h_sec": row["h_sec"],

                    "v_deg": row["v_deg"],
                    "v_min": row["v_min"],
                    "v_sec": row["v_sec"],

                    "distance_m": distance_m,
                    "d": distance_mm,
                }

                measurements.append(measurement)

            except Exception as e:
                raise ValueError(f"CSV 第 {row_index} 行解析失败：{row}\n原因：{e}") from e

    if len(measurements) < 4:
        raise ValueError(f"有效测量点数量不足：{len(measurements)}。拟合球面至少需要 4 个点。")

    return measurements


def total_station_to_xyz(measurements):
    """
    全站仪球坐标 → 笛卡尔坐标（以全站仪为原点）

    坐标系定义：
      X = d · cos(el) · sin(az)   东向
      Y = d · cos(el) · cos(az)   正前方
      Z = d · sin(el)             向上

    d 单位：mm
    """
    pts = []

    for m in measurements:
        az_deg = dms_to_deg(m["h_deg"], m["h_min"], m["h_sec"])
        el_deg = dms_to_deg(m["v_deg"], m["v_min"], m["v_sec"])

        az = np.radians(az_deg)
        el = np.radians(el_deg)

        d = float(m["d"])

        X = d * np.cos(el) * np.sin(az)
        Y = d * np.cos(el) * np.cos(az)
        Z = d * np.sin(el)

        pts.append([X, Y, Z])

    return np.array(pts, dtype=float)


def fit_sphere(pts):
    """
    最小二乘拟合球面：
      (X-Cx)^2 + (Y-Cy)^2 + (Z-Cz)^2 = r^2
    """
    A = np.column_stack([
        2 * pts[:, 0],
        2 * pts[:, 1],
        2 * pts[:, 2],
        np.ones(len(pts)),
    ])

    rhs = pts[:, 0] ** 2 + pts[:, 1] ** 2 + pts[:, 2] ** 2

    result, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)

    Cx, Cy, Cz = result[:3]
    b = result[3]

    radius_sq = b + Cx ** 2 + Cy ** 2 + Cz ** 2

    if radius_sq <= 0:
        raise ValueError(f"拟合得到的半径平方异常：{radius_sq}")

    r = np.sqrt(radius_sq)

    return np.array([Cx, Cy, Cz]), r


def build_result(measurements, pts, center, radius):
    """
    生成输出 JSON 内容。
    """
    dists = np.linalg.norm(pts - center, axis=1)
    residuals = dists - radius

    rms = float(np.sqrt(np.mean(residuals ** 2)))
    max_residual = float(np.max(np.abs(residuals)))

    point_results = []

    for i, m in enumerate(measurements):
        point_results.append({
            "index": i + 1,
            "point_id": m.get("point_id", ""),
            "image_name": m.get("image_name", ""),
            "xyz_mm": [
                float(pts[i, 0]),
                float(pts[i, 1]),
                float(pts[i, 2]),
            ],
            "residual_mm": float(residuals[i]),
            "distance_m": float(m["distance_m"]),
        })

    result = {
        "sphere_center": [
            float(center[0]),
            float(center[1]),
            float(center[2]),
        ],
        "sphere_radius": float(radius),
        "rms_residual_mm": rms,
        "max_residual_mm": max_residual,
        "num_points": len(pts),
        "coordinate_system": "全站仪坐标系（X=东，Y=正前方，Z=天顶）",
        "point_results": point_results,
        "points_xyz": pts.tolist(),
        "residuals_mm": residuals.tolist(),
    }

    return result


def print_result(measurements, pts, result):
    """
    在命令行打印结果。
    """
    print("=" * 70)
    print("全站仪 CSV 数据 → 球面参数拟合")
    print("=" * 70)

    print(f"\n有效点数量：{result['num_points']}")

    print("\n各标定点笛卡尔坐标 (mm)：")
    print(f"{'序号':>4}  {'point_id':>10}  {'image_name':>16}  {'X':>12}  {'Y':>12}  {'Z':>12}")

    for i, m in enumerate(measurements):
        print(
            f"{i + 1:4d}  "
            f"{str(m.get('point_id', '')):>10}  "
            f"{str(m.get('image_name', '')):>16}  "
            f"{pts[i, 0]:12.2f}  "
            f"{pts[i, 1]:12.2f}  "
            f"{pts[i, 2]:12.2f}"
        )

    Cx, Cy, Cz = result["sphere_center"]
    radius = result["sphere_radius"]

    print("\n拟合结果：")
    print(f"  球心：({Cx:.3f}, {Cy:.3f}, {Cz:.3f}) mm")
    print(f"  半径：{radius:.3f} mm")

    print("\n各点残差 (mm)：")
    for p in result["point_results"]:
        print(
            f"  {p['index']:2d}  "
            f"{p['point_id']:>10}  "
            f"{p['image_name']:>16}  "
            f"{p['residual_mm']:+.4f} mm"
        )

    print(f"\n  RMS 残差：{result['rms_residual_mm']:.4f} mm")
    print(f"  最大残差：{result['max_residual_mm']:.4f} mm")

    if result["rms_residual_mm"] < 5.0:
        print("  [OK] 拟合质量良好（RMS < 5mm）")
    else:
        print("  [WARN] 拟合残差偏大，请检查测量数据")


def parse_args():
    parser = argparse.ArgumentParser(
        description="从全站仪 CSV 数据拟合球幕球面参数"
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="输入 CSV 文件路径，例如 D:\\projects\\Hand_DR\\input\\single_spot_manifest.csv",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="sphere_params.json",
        help="输出 JSON 文件路径，默认：sphere_params.json",
    )

    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="包含 enabled 不为 1 的点。默认只使用 enabled=1 的点。",
    )

    parser.add_argument(
        "--print-json",
        action="store_true",
        help="在命令行额外打印完整 JSON 结果。",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    measurements = load_measurements_from_csv(
        input_path,
        enabled_only=not args.include_disabled,
    )

    pts = total_station_to_xyz(measurements)
    center, radius = fit_sphere(pts)

    result = build_result(measurements, pts, center, radius)

    print_result(measurements, pts, result)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存至：{output_path.resolve()}")

    if args.print_json:
        print("\n完整 JSON 结果：")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    return result


if __name__ == "__main__":
    main()
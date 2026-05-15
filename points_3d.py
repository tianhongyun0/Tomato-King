import argparse
import csv
import math
from pathlib import Path


def dms_to_deg(deg, minute, second):
    """
    度分秒转十进制度。
    例如：
    -32, 08, 16 -> -32.137777...
    24, 51, 37 -> 24.860277...
    """
    deg = float(deg)
    minute = float(minute)
    second = float(second)

    sign = -1.0 if deg < 0 else 1.0
    return sign * (abs(deg) + minute / 60.0 + second / 3600.0)

#主要是计算x,y,z坐标的，输入是水平角、垂直角和距离，输出是三维坐标
def measurement_to_xyz(
    h_deg,
    h_min,
    h_sec,
    v_deg,
    v_min,
    v_sec,
    distance,
    vertical_type="elevation",
    horizontal_type="instrument_zero",
):
    """
    根据水平角、垂直角、距离计算三维坐标。

    vertical_type:
        elevation : v 表示仰角/俯角，水平面为 0°，向上为正，向下为负
        zenith    : v 表示天顶角，从竖直方向开始量

    horizontal_type:
        instrument_zero : 全站仪常用形式，0°方向作为 Y 正方向
        math_azimuth    : 数学常用形式，0°方向作为 X 正方向
    """
    h = dms_to_deg(h_deg, h_min, h_sec)
    v = dms_to_deg(v_deg, v_min, v_sec)

    if vertical_type == "elevation":
        elevation_deg = v
    elif vertical_type == "zenith":
        elevation_deg = 90.0 - v
    else:
        raise ValueError("vertical_type must be 'elevation' or 'zenith'")

    az = math.radians(h)
    el = math.radians(elevation_deg)

    r_xy = distance * math.cos(el)

    if horizontal_type == "instrument_zero":
        # 0° 指向 Y 正方向
        x = r_xy * math.sin(az)
        y = r_xy * math.cos(az)
    elif horizontal_type == "math_azimuth":
        # 0° 指向 X 正方向
        x = r_xy * math.cos(az)
        y = r_xy * math.sin(az)
    else:
        raise ValueError("horizontal_type must be 'instrument_zero' or 'math_azimuth'")

    z = distance * math.sin(el)

    return x, y, z, h, elevation_deg


def is_enabled(value):
    if value is None:
        return True
    value = str(value).strip().lower()
    return value not in {"0", "false", "no", "off", "disabled"}


def convert_csv(
    input_csv,
    output_csv,
    vertical_type="elevation",
    horizontal_type="instrument_zero",
    output_unit="m",
    include_disabled=False,
):
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    rows_out = []

    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required_cols = {
            "image_name",
            "h_deg", "h_min", "h_sec",
            "v_deg", "v_min", "v_sec",
            "enabled",
            "point_id",
        }

        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV 缺少必要列: {sorted(missing)}")

        has_distance_m = "distance_m" in reader.fieldnames
        has_distance_mm = "distance_mm" in reader.fieldnames

        if not has_distance_m and not has_distance_mm:
            raise ValueError("CSV 中必须包含 distance_m 或 distance_mm")

        for row in reader:
            if not include_disabled and not is_enabled(row.get("enabled")):
                continue

            if has_distance_m and row.get("distance_m", "").strip() != "":
                distance_m = float(row["distance_m"])
            else:
                distance_m = float(row["distance_mm"]) / 1000.0

            if output_unit == "mm":
                distance = distance_m * 1000.0
            elif output_unit == "m":
                distance = distance_m
            else:
                raise ValueError("output_unit must be 'mm' or 'm'")

            x, y, z, h_decimal_deg, elevation_decimal_deg = measurement_to_xyz(
                row["h_deg"],
                row["h_min"],
                row["h_sec"],
                row["v_deg"],
                row["v_min"],
                row["v_sec"],
                distance,
                vertical_type=vertical_type,
                horizontal_type=horizontal_type,
            )

            row_out = dict(row)
            row_out["h_decimal_deg"] = f"{h_decimal_deg:.10f}"
            row_out["elevation_decimal_deg"] = f"{elevation_decimal_deg:.10f}"
            row_out[f"x_{output_unit}"] = f"{x:.6f}"
            row_out[f"y_{output_unit}"] = f"{y:.6f}"
            row_out[f"z_{output_unit}"] = f"{z:.6f}"

            rows_out.append(row_out)

    if not rows_out:
        raise RuntimeError("没有可输出的数据，请检查 enabled 是否全为 0")

    fieldnames = list(rows_out[0].keys())

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"转换完成，共输出 {len(rows_out)} 个点")
    print(f"输出文件: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="读取全站仪角度距离 CSV，并计算每个点的 3D 坐标"
    )

    parser.add_argument(
        "input_csv",
        help="输入 CSV 文件路径，例如 manifest.csv"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="points_3d.csv",
        help="输出 CSV 文件路径，默认 points_3d.csv"
    )

    parser.add_argument(
        "--vertical",
        choices=["elevation", "zenith"],
        default="elevation",
        help="垂直角类型：elevation=仰角/俯角，zenith=天顶角。默认 elevation"
    )

    parser.add_argument(
        "--horizontal",
        choices=["instrument_zero", "math_azimuth"],
        default="instrument_zero",
        help="水平角坐标约定。默认 instrument_zero"
    )

    parser.add_argument(
        "--unit",
        choices=["mm", "m"],
        default="m",
        help="输出坐标单位，默认 mm"
    )

    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="是否也输出 enabled=0 的点"
    )

    args = parser.parse_args()

    convert_csv(
        input_csv=args.input_csv,
        output_csv=args.output,
        vertical_type=args.vertical,
        horizontal_type=args.horizontal,
        output_unit=args.unit,
        include_disabled=args.include_disabled,
    )


if __name__ == "__main__":
    main()
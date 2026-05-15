import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def normalize_name(name: str) -> str:
    """
    用于匹配 CSV 里的 image_name 和 JSONL 里的 image_basename。
    """
    return Path(str(name).replace("\\", "/")).name.lower().strip()

def load_total_station_points(csv_path: str) -> pd.DataFrame:
    """
    读取全站仪 CSV。
    优先使用 x_m, y_m, z_m 三列作为 3D 点。
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "image_name" not in df.columns:
        raise ValueError("CSV 中必须包含 image_name 列，用于和图像中心结果匹配。")

    if "enabled" in df.columns:
        df = df[df["enabled"].astype(int) == 1].copy()

    required_cols = ["x_m", "y_m", "z_m"]
    if not all(c in df.columns for c in required_cols):
        raise ValueError("CSV 中必须包含 x_m, y_m, z_m 三列。")

    df["match_name"] = df["image_name"].apply(normalize_name)

    keep_cols = ["image_name", "match_name", "x_m", "y_m", "z_m"]
    if "point_id" in df.columns:
        keep_cols.insert(1, "point_id")

    return df[keep_cols].copy()


def load_center_results(center_path: str, pixel_key: str = "final_center_xy") -> pd.DataFrame:
    """
    读取中心点结果。
    支持：
    1. JSONL：一行一个 JSON
    2. JSON：列表格式
    """
    path = Path(center_path)
    records = []

    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[警告] 第 {line_idx} 行 JSON 解析失败，已跳过：{e}")
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            if "results" in data and isinstance(data["results"], list):
                records = data["results"]
            else:
                records = [data]
        else:
            raise ValueError("中心点结果 JSON 格式不支持。")

    rows = []
    for r in records:
        if pixel_key not in r:
            continue

        xy = r[pixel_key]
        if not isinstance(xy, (list, tuple)) or len(xy) < 2:
            continue

        image_name = (
            r.get("image_basename")
            or r.get("image_name")
            or r.get("image_path")
            or ""
        )

        if not image_name:
            continue

        rows.append({
            "center_image_name": image_name,
            "match_name": normalize_name(image_name),
            "u": float(xy[0]),
            "v": float(xy[1]),
        })

    if not rows:
        raise ValueError(f"没有从 {center_path} 中读取到有效的 {pixel_key} 数据。")

    df = pd.DataFrame(rows)

    # 如果同一张图重复出现，保留第一条
    before = len(df)
    df = df.drop_duplicates(subset=["match_name"], keep="first").copy()
    after = len(df)
    if before != after:
        print(f"[警告] 中心点结果中存在重复图像名，已从 {before} 条去重为 {after} 条。")

    return df


def load_intrinsics(json_path: str):
    """
    读取相机内参 JSON。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    K = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist = np.asarray(data["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)

    image_width = data.get("image_width", None)
    image_height = data.get("image_height", None)

    return K, dist, image_width, image_height, data


def make_matched_points(points_df: pd.DataFrame, centers_df: pd.DataFrame) -> pd.DataFrame:
    """
    按图像文件名匹配 3D 点和 2D 点。
    """
    matched = pd.merge(points_df, centers_df, on="match_name", how="inner")

    if len(matched) < 4:
        raise ValueError(
            f"有效匹配点数量只有 {len(matched)} 个，PnP 至少需要 4 个点，建议 6 个以上。"
        )

    all_3d_names = set(points_df["match_name"])
    all_2d_names = set(centers_df["match_name"])

    missing_2d = sorted(all_3d_names - all_2d_names)
    missing_3d = sorted(all_2d_names - all_3d_names)

    if missing_2d:
        print("\n[警告] 以下 CSV 中的图像没有找到对应像素中心：")
        for n in missing_2d:
            print("  -", n)

    if missing_3d:
        print("\n[提示] 以下中心点结果没有对应的全站仪 3D 点，已忽略：")
        for n in missing_3d:
            print("  -", n)

    return matched.reset_index(drop=True)


def get_pnp_flag(flag_name: str):
    """
    根据字符串获取 OpenCV solvePnP flag。
    """
    name = flag_name.upper()

    flag_map = {
        "ITERATIVE": cv2.SOLVEPNP_ITERATIVE,
        "EPNP": cv2.SOLVEPNP_EPNP,
        "P3P": cv2.SOLVEPNP_P3P,
        "AP3P": cv2.SOLVEPNP_AP3P,
    }

    if hasattr(cv2, "SOLVEPNP_SQPNP"):
        flag_map["SQPNP"] = cv2.SOLVEPNP_SQPNP

    if name not in flag_map:
        raise ValueError(f"不支持的 PnP 方法：{flag_name}，可选：{list(flag_map.keys())}")

    return flag_map[name], name


def solve_pose(
    object_points,
    image_points,
    K,
    dist,
    pnp_flag_name="ITERATIVE",
    refine=True,
    refine_max_iter=100,
    refine_eps=1e-10,
):
    """
    先 solvePnP，再 solvePnPRefineVVS。
    """
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)

    n = len(object_points)
    requested_flag, requested_name = get_pnp_flag(pnp_flag_name)

    # 对留一点验证来说，如果总共 6 个点，训练集只有 5 个点。
    # 某些 OpenCV 版本里 ITERATIVE 对少点初始化可能不稳，所以这里做备用方案。
    candidate_flags = [(requested_flag, requested_name)]

    if n < 6:
        if hasattr(cv2, "SOLVEPNP_SQPNP"):
            candidate_flags.append((cv2.SOLVEPNP_SQPNP, "SQPNP"))
        candidate_flags.append((cv2.SOLVEPNP_EPNP, "EPNP"))

    if requested_name != "ITERATIVE":
        candidate_flags.append((cv2.SOLVEPNP_ITERATIVE, "ITERATIVE"))

    last_error = None
    ok = False
    rvec = None
    tvec = None
    used_name = None

    for flag, name in candidate_flags:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                K,
                dist,
                flags=flag
            )
            if ok:
                used_name = name
                break
        except cv2.error as e:
            last_error = e

    if not ok:
        raise RuntimeError(f"cv2.solvePnP 失败。最后错误：{last_error}")

    rvec_initial = rvec.copy()
    tvec_initial = tvec.copy()

    refined_ok = False

    if refine:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
            int(refine_max_iter),
            float(refine_eps)
        )

        try:
            result = cv2.solvePnPRefineVVS(
                object_points,
                image_points,
                K,
                dist,
                rvec,
                tvec,
                criteria
            )

            if isinstance(result, tuple) and len(result) == 2:
                rvec, tvec = result

            refined_ok = True

        except cv2.error as e:
            print(f"[警告] solvePnPRefineVVS 失败，将使用 solvePnP 初值。错误：{e}")

    return {
        "rvec": rvec,
        "tvec": tvec,
        "rvec_initial": rvec_initial,
        "tvec_initial": tvec_initial,
        "used_pnp_flag": used_name,
        "refined_ok": refined_ok,
    }


def project_and_error(object_points, image_points, K, dist, rvec, tvec):
    """
    计算重投影点和误差。
    """
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)

    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        K,
        dist
    )
    projected = projected.reshape(-1, 2)

    delta = projected - image_points
    err = np.linalg.norm(delta, axis=1)

    return projected, delta, err


def calc_metrics(errors):
    errors = np.asarray(errors, dtype=np.float64)

    return {
        "mean_px": float(np.mean(errors)),
        "rmse_px": float(np.sqrt(np.mean(errors ** 2))),
        "median_px": float(np.median(errors)),
        "max_px": float(np.max(errors)),
        "std_px": float(np.std(errors)),
    }

#这段主要是求一下相机在世界坐标下的位置
def rotation_info(rvec, tvec):
    """
    输出旋转矩阵、相机中心等外参信息。
    OpenCV 中：
        X_cam = R @ X_world + t
    相机中心在世界坐标系中：
        C_world = -R.T @ t
    """
    R, _ = cv2.Rodrigues(rvec)
    C_world = -R.T @ tvec

    return R, C_world


def dataframe_to_points(df):
    object_points = df[["x_m", "y_m", "z_m"]].to_numpy(dtype=np.float64)
    image_points = df[["u", "v"]].to_numpy(dtype=np.float64)
    return object_points, image_points


def run_all_points_estimation(matched, K, dist, args, out_dir: Path):
    """
    全部点参与计算，然后输出每个点的重投影误差。
    注意：这里每个点既参与了外参求解，也参与了误差统计，所以这是拟合误差，不是独立验证误差。
    """
    object_points, image_points = dataframe_to_points(matched)

    pose = solve_pose(
        object_points,
        image_points,
        K,
        dist,
        pnp_flag_name=args.pnp_flag,
        refine=True,
        refine_max_iter=args.refine_max_iter,
        refine_eps=args.refine_eps,
    )

    rvec = pose["rvec"]
    tvec = pose["tvec"]

    projected, delta, err = project_and_error(
        object_points,
        image_points,
        K,
        dist,
        rvec,
        tvec
    )

    metrics = calc_metrics(err)
    R, C_world = rotation_info(rvec, tvec)

    result_df = matched.copy()
    result_df["projected_u"] = projected[:, 0]
    result_df["projected_v"] = projected[:, 1]
    result_df["du_px"] = delta[:, 0]
    result_df["dv_px"] = delta[:, 1]
    result_df["reprojection_error_px"] = err

    result_csv = out_dir / "all_points_reprojection_errors.csv"
    result_df.to_csv(result_csv, index=False, encoding="utf-8-sig")

    pose_json = {
        "method": "all_points",
        "used_pnp_flag": pose["used_pnp_flag"],
        "refined_by_solvePnPRefineVVS": pose["refined_ok"],
        "num_points": int(len(matched)),
        "rvec": rvec.reshape(-1).tolist(),
        "tvec": tvec.reshape(-1).tolist(),
        "rotation_matrix_R": R.tolist(),
        "camera_center_world": C_world.reshape(-1).tolist(),
        "metrics": metrics,
    }

    pose_path = out_dir / "all_points_pose.json"
    with open(pose_path, "w", encoding="utf-8") as f:
        json.dump(pose_json, f, ensure_ascii=False, indent=2)

    print("\n================ 全部点参与计算 ================")
    print(f"匹配点数量：{len(matched)}")
    print(f"solvePnP 使用方法：{pose['used_pnp_flag']}")
    print(f"RefineVVS 是否成功：{pose['refined_ok']}")

    print("\n旋转向量 rvec：")
    print(rvec.reshape(-1))

    print("\n平移向量 tvec：")
    print(tvec.reshape(-1))

    print("\n旋转矩阵 R：")
    print(R)

    print("\n相机中心在全站仪/世界坐标系下的位置 C_world = -R.T @ t：")
    print(C_world.reshape(-1))

    print("\n---------------- 每个点的重投影误差：全部点参与计算 ----------------")
    print("说明：这里的误差是拟合误差，因为这些点都参与了外参求解。")

    display_cols = []

    if "point_id" in result_df.columns:
        display_cols.append("point_id")

    display_cols += [
        "image_name",
        "x_m", "y_m", "z_m",
        "u", "v",
        "projected_u", "projected_v",
        "du_px", "dv_px",
        "reprojection_error_px"
    ]

    display_df = result_df[display_cols].copy()

    round_cols = [
        "x_m", "y_m", "z_m",
        "u", "v",
        "projected_u", "projected_v",
        "du_px", "dv_px",
        "reprojection_error_px"
    ]

    for col in round_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].astype(float).round(6)

    print(display_df.to_string(index=False))

    print("\n---------------- 全部点参与计算：重投影误差评价 ----------------")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    print(f"\n逐点重投影误差已保存：{result_csv}")
    print(f"外参结果已保存：{pose_path}")

    return pose_json, result_df


def run_leave_one_out(matched, K, dist, args, out_dir: Path):
    """
    留一点验证：
    每次拿 N-1 个点求外参，剩下 1 个点只用于验证。

    holdout_error_px 才是该留出点的独立验证误差。
    train_mean_error_px / train_rmse_error_px 是剩余训练点的拟合误差。
    """
    if len(matched) < 5:
        print("\n[警告] 点数少于 5，不适合做留一点验证，已跳过。")
        return None

    rows = []

    print("\n================ 留一点验证 Leave-One-Out ================")
    print("说明：每一行表示：该点不参与外参求解，只作为验证点。")
    print("holdout_error_px 表示这个被留出点的验证误差。\n")

    for hold_idx in range(len(matched)):
        train_df = matched.drop(index=hold_idx).reset_index(drop=True)
        test_df = matched.iloc[[hold_idx]].reset_index(drop=True)

        obj_train, img_train = dataframe_to_points(train_df)
        obj_test, img_test = dataframe_to_points(test_df)

        try:
            pose = solve_pose(
                obj_train,
                img_train,
                K,
                dist,
                pnp_flag_name=args.pnp_flag,
                refine=True,
                refine_max_iter=args.refine_max_iter,
                refine_eps=args.refine_eps,
            )
        except Exception as e:
            print(f"[警告] 留出 {test_df.loc[0, 'image_name']} 时求解失败：{e}")
            continue

        projected_test, delta_test, err_test = project_and_error(
            obj_test,
            img_test,
            K,
            dist,
            pose["rvec"],
            pose["tvec"]
        )

        projected_train, delta_train, err_train = project_and_error(
            obj_train,
            img_train,
            K,
            dist,
            pose["rvec"],
            pose["tvec"]
        )

        train_metrics = calc_metrics(err_train)

        row = {
            "holdout_index": int(hold_idx),
            "holdout_image_name": test_df.loc[0, "image_name"],
            "holdout_match_name": test_df.loc[0, "match_name"],
            "used_pnp_flag": pose["used_pnp_flag"],
            "refined_ok": pose["refined_ok"],

            "X_m": float(obj_test[0, 0]),
            "Y_m": float(obj_test[0, 1]),
            "Z_m": float(obj_test[0, 2]),

            "measured_u": float(img_test[0, 0]),
            "measured_v": float(img_test[0, 1]),
            "projected_u": float(projected_test[0, 0]),
            "projected_v": float(projected_test[0, 1]),

            "du_px": float(delta_test[0, 0]),
            "dv_px": float(delta_test[0, 1]),
            "holdout_error_px": float(err_test[0]),

            "train_mean_error_px": train_metrics["mean_px"],
            "train_rmse_error_px": train_metrics["rmse_px"],
            "train_max_error_px": train_metrics["max_px"],
        }

        if "point_id" in test_df.columns:
            row["point_id"] = test_df.loc[0, "point_id"]

        rows.append(row)

        point_label = row.get("point_id", f"index_{hold_idx}")
        print(
            f"留出点：{point_label}, "
            f"图像：{row['holdout_image_name']}, "
            f"验证误差 holdout_error_px = {row['holdout_error_px']:.6f} px"
        )

    if not rows:
        print("\n[警告] 留一点验证没有成功结果。")
        return None

    loo_df = pd.DataFrame(rows)

    loo_csv = out_dir / "leave_one_out_validation.csv"
    loo_df.to_csv(loo_csv, index=False, encoding="utf-8-sig")

    metrics = calc_metrics(loo_df["holdout_error_px"].to_numpy())

    summary = {
        "method": "leave_one_out",
        "num_success": int(len(loo_df)),
        "metrics": metrics,
    }

    summary_path = out_dir / "leave_one_out_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n---------------- 每个点的留一点验证误差 ----------------")
    print("说明：每一行都是该点被留出时的独立验证误差。")

    display_cols = []

    if "point_id" in loo_df.columns:
        display_cols.append("point_id")

    display_cols += [
        "holdout_image_name",
        "X_m", "Y_m", "Z_m",
        "measured_u", "measured_v",
        "projected_u", "projected_v",
        "du_px", "dv_px",
        "holdout_error_px",
        "train_mean_error_px",
        "train_rmse_error_px",
        "train_max_error_px",
        "used_pnp_flag",
        "refined_ok"
    ]

    display_df = loo_df[display_cols].copy()

    round_cols = [
        "X_m", "Y_m", "Z_m",
        "measured_u", "measured_v",
        "projected_u", "projected_v",
        "du_px", "dv_px",
        "holdout_error_px",
        "train_mean_error_px",
        "train_rmse_error_px",
        "train_max_error_px"
    ]

    for col in round_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].astype(float).round(6)

    print(display_df.to_string(index=False))

    print("\n---------------- 留一点验证：整体误差评价 ----------------")
    print(f"成功验证次数：{len(loo_df)}")

    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    print(f"\n留一点验证逐点结果已保存：{loo_csv}")
    print(f"留一点验证汇总已保存：{summary_path}")

    return summary, loo_df

def remove_worst_point_and_reestimate(
    matched,
    K,
    dist,
    args,
    out_dir: Path,
    loo_df=None,
    all_points_error_df=None,
    remove_by="loo"
):
    """
    剔除误差最大的一个点后，重新计算外参，并重新输出：
    1. 全部点参与计算误差评价
    2. 留一点验证误差评价

    remove_by:
        "loo"：根据留一点验证的 holdout_error_px 最大值剔除，推荐
        "all"：根据全部点参与计算的 reprojection_error_px 最大值剔除
    """

    print("\n\n================ 剔除最大误差点后重新计算 ================")

    if len(matched) <= 5:
        print(
            f"[提示] 当前原始匹配点数量为 {len(matched)}。"
            "剔除 1 个点后只剩 4 个或更少点，不建议继续做稳定外参估计。"
        )

    remove_by = remove_by.lower().strip()

    if remove_by == "loo":
        if loo_df is None or len(loo_df) == 0:
            print("[警告] 没有留一点验证结果，无法按 holdout_error_px 剔除最大误差点。")
            return None

        worst_idx = loo_df["holdout_error_px"].astype(float).idxmax()
        worst_row = loo_df.loc[worst_idx]

        remove_match_name = worst_row["holdout_match_name"]
        remove_error = float(worst_row["holdout_error_px"])
        remove_error_name = "holdout_error_px"

        print("\n剔除依据：留一点验证误差最大")
        print(f"最大误差字段：{remove_error_name}")

    elif remove_by == "all":
        if all_points_error_df is None or len(all_points_error_df) == 0:
            print("[警告] 没有全部点误差结果，无法按 reprojection_error_px 剔除最大误差点。")
            return None

        worst_idx = all_points_error_df["reprojection_error_px"].astype(float).idxmax()
        worst_row = all_points_error_df.loc[worst_idx]

        remove_match_name = worst_row["match_name"]
        remove_error = float(worst_row["reprojection_error_px"])
        remove_error_name = "reprojection_error_px"

        print("\n剔除依据：全部点参与计算时的重投影误差最大")
        print(f"最大误差字段：{remove_error_name}")

    else:
        raise ValueError("remove_by 只能是 'loo' 或 'all'。")

    remove_rows = matched[matched["match_name"] == remove_match_name]

    if len(remove_rows) == 0:
        print(f"[警告] 在 matched 中没有找到需要剔除的点：{remove_match_name}")
        return None

    remove_row = remove_rows.iloc[0]

    remove_point_id = remove_row["point_id"] if "point_id" in remove_row.index else ""
    remove_image_name = remove_row["image_name"]

    print("\n被剔除的最大误差点：")
    if remove_point_id != "":
        print(f"  point_id: {remove_point_id}")
    print(f"  image_name: {remove_image_name}")
    print(f"  match_name: {remove_match_name}")
    print(f"  {remove_error_name}: {remove_error:.6f} px")
    print(
        f"  3D坐标: "
        f"X={float(remove_row['x_m']):.6f}, "
        f"Y={float(remove_row['y_m']):.6f}, "
        f"Z={float(remove_row['z_m']):.6f}"
    )
    print(
        f"  2D像素: "
        f"u={float(remove_row['u']):.6f}, "
        f"v={float(remove_row['v']):.6f}"
    )

    # 剔除最大误差点
    matched_removed = matched[matched["match_name"] != remove_match_name].reset_index(drop=True)

    print(f"\n剔除前点数：{len(matched)}")
    print(f"剔除后点数：{len(matched_removed)}")

    if len(matched_removed) < 4:
        print("[错误] 剔除后点数少于 4，无法继续 solvePnP。")
        return None

    # 单独建一个输出目录，避免和原始结果混在一起
    remove_out_dir = out_dir / f"remove_worst_by_{remove_by}"
    remove_out_dir.mkdir(parents=True, exist_ok=True)

    removed_csv = remove_out_dir / "matched_after_remove_worst.csv"
    matched_removed.to_csv(removed_csv, index=False, encoding="utf-8-sig")

    removed_info = {
        "remove_by": remove_by,
        "remove_error_name": remove_error_name,
        "remove_error_px": remove_error,
        "removed_point": {
            "point_id": str(remove_point_id),
            "image_name": str(remove_image_name),
            "match_name": str(remove_match_name),
            "x_m": float(remove_row["x_m"]),
            "y_m": float(remove_row["y_m"]),
            "z_m": float(remove_row["z_m"]),
            "u": float(remove_row["u"]),
            "v": float(remove_row["v"]),
        },
        "num_points_before": int(len(matched)),
        "num_points_after": int(len(matched_removed)),
    }

    removed_info_path = remove_out_dir / "removed_worst_point_info.json"
    with open(removed_info_path, "w", encoding="utf-8") as f:
        json.dump(removed_info, f, ensure_ascii=False, indent=2)

    print(f"\n剔除后的匹配点已保存：{removed_csv}")
    print(f"被剔除点信息已保存：{removed_info_path}")

    print("\n\n******** 剔除最大误差点后：全部点参与计算 ********")
    all_pose_removed, all_error_removed_df = run_all_points_estimation(
        matched_removed,
        K,
        dist,
        args,
        remove_out_dir
    )

    print("\n\n******** 剔除最大误差点后：留一点验证 ********")
    loo_removed_result = None

    if args.holdout == "loo":
        loo_removed_result = run_leave_one_out(
            matched_removed,
            K,
            dist,
            args,
            remove_out_dir
        )
    else:
        print("[提示] 当前 holdout 设置为 none，不进行留一点验证。")

    return {
        "removed_info": removed_info,
        "matched_removed": matched_removed,
        "all_pose_removed": all_pose_removed,
        "all_error_removed_df": all_error_removed_df,
        "loo_removed_result": loo_removed_result,
        "remove_out_dir": str(remove_out_dir),
    }

def main():
    parser = argparse.ArgumentParser(
        description="使用全站仪 3D 点 + 图像 2D 中心点进行相机外参标定：solvePnP + solvePnPRefineVVS"
    )

    parser.add_argument("--points_csv", required=True, help="全站仪 3D 点 CSV 路径")
    parser.add_argument("--centers_json", required=True, help="图像中心点 JSON/JSONL 路径")
    parser.add_argument("--intrinsics_json", required=True, help="相机内参 JSON 路径")
    parser.add_argument("--out_dir", default="extrinsic_result", help="输出目录")

    parser.add_argument(
        "--pixel_key",
        default="final_center_xy",
        help="中心点结果中使用的字段名，默认 final_center_xy"
    )

    parser.add_argument(
        "--pnp_flag",
        default="ITERATIVE",
        choices=["ITERATIVE", "EPNP", "SQPNP", "P3P", "AP3P"],
        help="solvePnP 使用的方法，默认 ITERATIVE"
    )

    parser.add_argument(
        "--holdout",
        default="loo",
        choices=["loo", "none"],
        help="验证方式：loo 表示 Leave-One-Out；none 表示不做留一点验证"
    )
    parser.add_argument(
    "--remove_worst",
    action="store_true",
    help="是否剔除误差最大的一个点后重新计算外参"
)

    parser.add_argument(
        "--remove_by",
        default="loo",
        choices=["loo", "all"],
        help="剔除最大误差点的依据：loo 表示按留一点验证误差 holdout_error_px；all 表示按全部点拟合误差 reprojection_error_px"
    )

    parser.add_argument("--refine_max_iter", type=int, default=100)
    parser.add_argument("--refine_eps", type=float, default=1e-10)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n读取全站仪 3D 点...")
    points_df = load_total_station_points(args.points_csv)

    print("读取图像 2D 中心点...")
    centers_df = load_center_results(args.centers_json, pixel_key=args.pixel_key)

    print("读取相机内参...")
    K, dist, image_width, image_height, intrinsics_raw = load_intrinsics(args.intrinsics_json)

    print("\n相机内参 K：")
    print(K)

    print("\n畸变系数 dist：")
    print(dist.reshape(-1))

    if image_width is not None and image_height is not None:
        print(f"\n图像尺寸：{image_width} x {image_height}")

    print("\n匹配 3D-2D 点...")
    matched = make_matched_points(points_df, centers_df)

    matched_csv = out_dir / "matched_3d_2d_points.csv"
    matched.to_csv(matched_csv, index=False, encoding="utf-8-sig")

    print(f"\n有效匹配点数量：{len(matched)}")
    print(f"匹配后的 3D-2D 点已保存：{matched_csv}")

    print("\n匹配点预览：")
    preview_cols = ["image_name", "x_m", "y_m", "z_m", "u", "v"]
    if "point_id" in matched.columns:
        preview_cols.insert(1, "point_id")
    print(matched[preview_cols].to_string(index=False))

    # 全部点参与计算
    all_pose, all_error_df = run_all_points_estimation(
        matched,
        K,
        dist,
        args,
        out_dir
    )

    # 留一点验证
    loo_result = None
    loo_df = None

    if args.holdout == "loo":
        loo_result = run_leave_one_out(
            matched,
            K,
            dist,
            args,
            out_dir
        )

        if loo_result is not None:
            loo_summary, loo_df = loo_result

    # 剔除最大误差点后重新计算
    if args.remove_worst:
        remove_worst_point_and_reestimate(
            matched=matched,
            K=K,
            dist=dist,
            args=args,
            out_dir=out_dir,
            loo_df=loo_df,
            all_points_error_df=all_error_df,
            remove_by=args.remove_by
        )

    print("\n全部完成。")

    print("\n全部完成。")


if __name__ == "__main__":
    main()
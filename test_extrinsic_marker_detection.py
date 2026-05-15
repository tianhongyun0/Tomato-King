"""
命令行测试外参红点检测（逻辑与 pipeline_api.calibrate_camera_extrinsics 内联段对齐，
实现见 lib/marker_detection_extrinsics.py；pipeline_api 不依赖该文件）。

不写 camera_extrinsics.json，仅输出调试图与终端日志。

用法:
  python scripts/test_extrinsic_marker_detection.py --image D:/path/to/photo.jpg
  python scripts/test_extrinsic_marker_detection.py --image photo.jpg --sphere-params data/calibration/sphere_params.json
  python scripts/test_extrinsic_marker_detection.py --image photo.jpg -o test_output/my_run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2  # noqa: E402

from lib.marker_detection_extrinsics import (  # noqa: E402
    detect_marker_centers_from_bgr,
    image_grid_splits_from_candidates,
    save_masks_and_all_markers,
    save_selected_and_grid_match,
    select_nine_by_pixel_variance,
    world_image_grid_match,
)


def main() -> int:
    p = argparse.ArgumentParser(description="测试外参标定红点检测（pipeline 同款逻辑）")
    p.add_argument("--image", required=True, help="输入 BGR 图像路径")
    p.add_argument(
        "--sphere-params",
        default=None,
        help="含 points_xyz 的 JSON（如 sphere_params.json）；不提供则只做检测与图像网格线，不做世界点匹配",
    )
    p.add_argument(
        "-o",
        "--out-dir",
        default=None,
        help="调试图输出目录（默认: test_output/marker_detection/<图片主文件名>）",
    )
    args = p.parse_args()

    image_path = os.path.abspath(args.image)
    if not os.path.isfile(image_path):
        print(f"错误: 找不到文件 {image_path}")
        return 1

    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = args.out_dir or os.path.join(_ROOT, "test_output", "marker_detection", stem)
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: OpenCV 无法读取图像: {image_path}")
        return 1

    print(f"图像: {image_path}  尺寸 {img.shape[1]}x{img.shape[0]}")
    print(f"输出目录: {out_dir}\n")

    centers, dbg = detect_marker_centers_from_bgr(img)
    save_masks_and_all_markers(img, centers, dbg, out_dir)

    if len(centers) < 9:
        print(f"\n失败: 候选点仅 {len(centers)} 个（需要 ≥9）。已保存掩码与候选可视化到上述目录。")
        return 2

    try:
        candidates, best_start, best_var = select_nine_by_pixel_variance(centers)
    except RuntimeError as e:
        print(f"\n失败: {e}")
        return 2

    print(f"\n[9 点窗口] 起始索引={best_start}, pixels 方差={best_var:.1f}")
    for i, c in enumerate(candidates):
        print(
            f"  #{i+1}: ({c['cx']:.1f}, {c['cy']:.1f})  "
            f"pixels={c['pixels']}  circularity={c.get('circularity', 0):.3f}  type={c['type']}"
        )

    pts3d = None
    if args.sphere_params:
        sp_path = args.sphere_params
        if not os.path.isabs(sp_path):
            sp_path = os.path.join(_ROOT, sp_path)
        if not os.path.isfile(sp_path):
            print(f"\n错误: 找不到 sphere-params 文件 {sp_path}")
            return 1
        with open(sp_path, encoding="utf-8") as f:
            sp = json.load(f)
        pts3d = np.array(sp["points_xyz"], dtype=np.float64)
        world_pts, image_pts, splits = world_image_grid_match(candidates, pts3d)
        print(f"\n[网格匹配] 与全站仪点对齐 {len(world_pts)} 对")
        print(
            f"  行分界线 y1={splits['y_split1']:.1f}, y2={splits['y_split2']:.1f}  "
            f"列分界线 x1={splits['x_split1']:.1f}, x2={splits['x_split2']:.1f}"
        )
        save_selected_and_grid_match(img, candidates, world_pts, image_pts, splits, out_dir)
    else:
        splits = image_grid_splits_from_candidates(candidates)
        image_pts = np.array([[c["cx"], c["cy"]] for c in candidates], dtype=np.float64)
        world_pts = np.zeros((0, 3), dtype=np.float64)
        print("\n[仅图像] 未提供 --sphere-params：绘制 9 点与图像侧 3×3 分割线（无世界坐标匹配）")
        print(
            f"  行分界线 y1={splits['y_split1']:.1f}, y2={splits['y_split2']:.1f}  "
            f"列分界线 x1={splits['x_split1']:.1f}, x2={splits['x_split2']:.1f}"
        )
        save_selected_and_grid_match(img, candidates, world_pts, image_pts, splits, out_dir)

    print("\n完成。生成文件（与 pipeline vis_dir 命名一致）:")
    for name in (
        "marker_detection_all.jpg",
        "mask_red_hsv.jpg",
        "mask_red_bgr.jpg",
        "mask_white.jpg",
        "mask_combined.jpg",
        "marker_detection_selected.jpg",
        "marker_grid_matching.jpg",
    ):
        fp = os.path.join(out_dir, name)
        if os.path.isfile(fp):
            print(f"  {fp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

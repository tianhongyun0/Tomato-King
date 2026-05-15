#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双目标定：从左右相机已采集的棋盘格图像对已知的左右内参，估计右相机相对左相机的旋转 R 与平移 T，
保存结果，并在一对样本图像上绘制 3D 坐标轴用于检查。

用法示例（两种等价）：
  python calib.py -w 9 -h 6 --square_size 10 \\
      ./data/left/ ./output/left/intrinsics.jsonl ./data/right/ ./output/right/intrinsics.jsonl \\
      -o ./output/stereo/

  python calib.py -w 9 -h 6 --square_size 10 \\
      --input ./data/left/ ./output/left/intrinsics.jsonl ./data/right/ ./output/right/intrinsics.jsonl \\
      --output ./output/stereo/

顺序说明：左图像目录 → 左 intrinsics.jsonl → 右图像目录 → 右 intrinsics.jsonl（使用 --input 时省略第一项）。

默认在 --output 目录写入：
  stereo_extrinsics.jsonl — 仅相对外参 R、T；
  stereo_calibration_meta.jsonl — RMS、E、F、分辨率、棋盘参数、左右内参路径、有效样本对数等；
  stereo_projection.json — 左/右内参与畸变、R/T、P=[R|T]、本质矩阵 E、基础矩阵 F；横行排版。

依赖：需与本项目 calibrate.py 相同环境（numpy、opencv-contrib-python）。
"""

from __future__ import print_function

import argparse
import json
import os
import sys

import cv2 as cv
import numpy as np

# 与 calibrate.py 共用图像列表展开逻辑
from calibrate import expand_image_inputs


def load_intrinsics_jsonl(path):
    """读取 calibrate.py 生成的 intrinsics.jsonl（单行 JSON）。"""
    path = os.path.normpath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError('找不到内参文件: %s' % path)
    with open(path, 'r', encoding='utf-8') as f:
        line = f.readline().strip()
    if not line:
        raise ValueError('内参文件为空: %s' % path)
    data = json.loads(line)
    k = np.asarray(data['camera_matrix'], dtype=np.float64)
    d = np.asarray(data['distortion_coefficients'], dtype=np.float64).reshape(-1, 1)
    return k, d


def build_object_points(pattern_size_hw, square_size):
    """棋盘格内角点对应的物体坐标，与 calibrate.py 一致（XY 平面，Z=0）。"""
    width, height = pattern_size_hw
    pts = np.zeros((width * height, 3), np.float32)
    pts[:, :2] = np.indices((width, height)).T.reshape(-1, 2).astype(np.float32)
    pts *= float(square_size)
    return pts


def stereo_calibrate_from_images(
    left_files,
    right_files,
    camera_matrix_left,
    dist_left,
    camera_matrix_right,
    dist_right,
    pattern_size_wh,
    square_size,
):
    """
    pattern_size_wh: findChessboardCorners 的 (columns, rows)，即内角点宽、高个数。
    返回 stereoCalibrate 的 ret, R, T, E, F 以及用于可视化的样本图与姿态。
    """
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    pattern_cols, pattern_rows = pattern_size_wh
    objp = build_object_points((pattern_cols, pattern_rows), square_size)

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    image_size = None

    if len(left_files) != len(right_files):
        print(
            '警告: 左右图像数量不一致 (%d vs %d)，将按较短一侧对齐。'
            % (len(left_files), len(right_files)),
            file=sys.stderr,
        )
    n_pairs = min(len(left_files), len(right_files))

    vis_left_path = None
    vis_right_path = None
    corners_left_vis = None
    corners_right_vis = None

    for i in range(n_pairs):
        im0 = cv.imread(left_files[i])
        im1 = cv.imread(right_files[i])
        if im0 is None or im1 is None:
            print('跳过无法读取的一对: %s | %s' % (left_files[i], right_files[i]), file=sys.stderr)
            continue
        if im0.shape[:2] != im1.shape[:2]:
            print(
                '跳过尺寸不一致的一对: %s vs %s'
                % (left_files[i], right_files[i]),
                file=sys.stderr,
            )
            continue

        gray0 = cv.cvtColor(im0, cv.COLOR_BGR2GRAY)
        gray1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
        ok0, corners0 = cv.findChessboardCorners(gray0, (pattern_cols, pattern_rows), None)
        ok1, corners1 = cv.findChessboardCorners(gray1, (pattern_cols, pattern_rows), None)
        if not ok0 or not ok1:
            continue

        corners0 = cv.cornerSubPix(gray0, corners0, (11, 11), (-1, -1), criteria)
        corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints_left.append(corners0)
        imgpoints_right.append(corners1)
        if image_size is None:
            image_size = (im0.shape[1], im0.shape[0])

        if vis_left_path is None:
            vis_left_path = left_files[i]
            vis_right_path = right_files[i]
            corners_left_vis = corners0.copy()
            corners_right_vis = corners1.copy()

    if len(objpoints) < 3:
        raise RuntimeError(
            '有效棋盘格样本过少（至少需要若干对；当前有效对数 %d）。请检查路径、棋盘尺寸参数与图像。'
            % len(objpoints)
        )

    if image_size is None:
        raise RuntimeError('未能确定图像尺寸。')
    flags = cv.CALIB_FIX_INTRINSIC
    ret, _, _, _, _, R, T, E, F = cv.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        camera_matrix_left,
        dist_left,
        camera_matrix_right,
        dist_right,
        image_size,
        criteria=criteria,
        flags=flags,
    )

    num_pairs = len(objpoints)
    return ret, R, T, E, F, vis_left_path, vis_right_path, corners_left_vis, corners_right_vis, objp, image_size, num_pairs


def draw_axes_pair(
    img_left_bgr,
    img_right_bgr,
    camera_matrix_left,
    dist_left,
    camera_matrix_right,
    dist_right,
    corners_left,
    objp,
    R_lr,
    T_lr,
    axis_length_factor,
):
    """
    以棋盘为世界坐标系，用左图 solvePnP 得到板相对左相机的外参；
    右图使用 R_W1 = R_lr @ R_W0, T_W1 = R_lr @ T_W0 + T_lr（与原版 calib.py 一致）。
    """
    ok, rvec0, tvec0 = cv.solvePnP(objp, corners_left, camera_matrix_left, dist_left)
    if not ok:
        return img_left_bgr, img_right_bgr

    R_W0, _ = cv.Rodrigues(rvec0)

    unit_axes = axis_length_factor * np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    ).reshape((4, 1, 3))

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    img0 = img_left_bgr.copy()
    img1 = img_right_bgr.copy()

    pts0, _ = cv.projectPoints(unit_axes, rvec0, tvec0, camera_matrix_left, dist_left)
    pts0 = pts0.reshape((4, 2)).astype(np.int32)
    o0 = tuple(pts0[0])
    for c, p in zip(colors, pts0[1:]):
        cv.line(img0, o0, tuple(p), c, 3)

    R_W1 = R_lr.dot(R_W0)
    tvec0_col = tvec0.reshape(3, 1)
    T_lr_col = T_lr.reshape(3, 1)
    tvec1 = R_lr.dot(tvec0_col) + T_lr_col
    rvec1, _ = cv.Rodrigues(R_W1)

    pts1, _ = cv.projectPoints(unit_axes, rvec1, tvec1, camera_matrix_right, dist_right)
    pts1 = pts1.reshape((4, 2)).astype(np.int32)
    o1 = tuple(pts1[0])
    for c, p in zip(colors, pts1[1:]):
        cv.line(img1, o1, tuple(p), c, 3)

    return img0, img1


def extrinsic_projection_matrix_RT(R, T):
    """外参组合的 3×4 矩阵 P = [R | T]（前 3 列为旋转，最后一列为平移）。"""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T = np.asarray(T, dtype=np.float64).reshape(3, 1)
    return np.hstack((R, T))


def write_projection_bundle_json_horizontal(path, bundle):
    """立体标定汇总 JSON：内参/畸变、R/T、P=[R|T]、E/F；矩阵横行排版。"""

    def dumps(o):
        return json.dumps(o, ensure_ascii=False)

    def write_matrix_block(f, key, rows):
        f.write('  "%s": [\n' % key)
        lines = []
        for i, row in enumerate(rows):
            sep = ',' if i < len(rows) - 1 else ''
            lines.append('    ' + dumps(row) + sep)
        f.write('\n'.join(lines))
        f.write('\n  ]')

    def write_shapes_section(f, b):
        f.write('  "shapes": {\n')
        pairs = []
        for name in ('K_left', 'K_right', 'R', 'P', 'E', 'F'):
            if name not in b:
                continue
            m = b[name]
            pairs.append('    "%s": [%d, %d]' % (name, len(m), len(m[0]) if m else 0))
        for name in ('distortion_coefficients_left', 'distortion_coefficients_right', 'T'):
            if name not in b:
                continue
            v = b[name]
            pairs.append('    "%s": [%d]' % (name, len(v)))
        f.write(',\n'.join(pairs))
        f.write('\n  },\n')

    key_order = [
        'convention',
        'shapes',
        'K_left',
        'distortion_coefficients_left',
        'K_right',
        'distortion_coefficients_right',
        'R',
        'T',
        'P',
        'E',
        'F',
    ]

    with open(path, 'w', encoding='utf-8') as f:
        f.write('{\n')
        f.write('  "convention": ')
        f.write(dumps(bundle['convention']))
        f.write(',\n')

        write_shapes_section(f, bundle)

        first = True
        for key in key_order:
            if key in ('convention', 'shapes'):
                continue
            if key not in bundle:
                continue
            if not first:
                f.write(',\n')
            first = False
            val = bundle[key]
            if key.startswith('distortion') or key == 'T':
                f.write('  "%s": ' % key)
                f.write(dumps(val))
            else:
                write_matrix_block(f, key, val)

        f.write('\n}\n')


def main():
    # 使用 -h 表示棋盘纵向内角点数，因此关闭默认 -h/--help，改用 --help
    parser = argparse.ArgumentParser(
        description='双目标定（外参）：估计左右相机之间的 R、T，并保存结果与坐标轴检查图。',
        add_help=False,
    )
    parser.add_argument('--help', action='help', help='显示帮助并退出')
    parser.add_argument('-w', type=int, default=9, metavar='W', help='棋盘横向内角点数（默认 9）')
    parser.add_argument(
        '-h',
        dest='pattern_h',
        type=int,
        default=6,
        metavar='H',
        help='棋盘纵向内角点数（默认 6）',
    )
    parser.add_argument('--square_size', type=float, default=10.0, help='方格边长，与单目标定一致（默认 10）')
    parser.add_argument('-o', '--output', default='./output/stereo', help='标定结果输出目录（默认 ./output/stereo）')
    parser.add_argument('--axis-scale', type=float, default=5.0, dest='axis_scale', help='坐标轴可视化长度比例（默认 5）')
    parser.add_argument(
        '--extrinsics-jsonl',
        default=None,
        metavar='PATH',
        help='仅含 R、T 的输出路径（默认 <输出目录>/stereo_extrinsics.jsonl）',
    )
    parser.add_argument(
        '--stereo-meta-jsonl',
        default=None,
        metavar='PATH',
        help='其余标定信息的输出路径（默认 <输出目录>/stereo_calibration_meta.jsonl）',
    )
    parser.add_argument(
        '--projection-json',
        default=None,
        metavar='PATH',
        help='汇总 JSON：左右内参与畸变、R/T、P=[R|T]、E/F（默认 <输出目录>/stereo_projection.json）',
    )
    parser.add_argument(
        '--input',
        default=None,
        metavar='LEFT_DIR',
        help='左相机图像目录或通配符；若指定，则后面只需再跟 3 个路径（左内参、右图像、右内参）',
    )
    parser.add_argument(
        'paths',
        nargs='*',
        metavar='PATH',
        help='无 --input：左图目录 | 左 intrinsics.jsonl | 右图目录 | 右 intrinsics.jsonl；'
        '有 --input：左 intrinsics | 右图目录 | 右 intrinsics',
    )

    ns, unknown = parser.parse_known_args()
    junk = {'.', './.', '.\\.', '.\\'}
    unknown = [u for u in unknown if u not in junk]
    if unknown:
        parser.error('无法解析的参数: %s' % ' '.join(unknown))

    pw = ns.w
    ph = ns.pattern_h
    square_size = ns.square_size
    out_dir = ns.output
    axis_scale = ns.axis_scale

    paths = list(ns.paths)
    # 忽略末尾单独的 "."（常见于复制命令时的占位）
    if paths and paths[-1] in ('.', './.', '.\\.'):
        paths.pop()

    if ns.input is not None:
        if len(paths) != 3:
            print(
                '错误：使用 --input 时需再提供恰好 3 个路径：左内参.jsonl、右图像目录、右内参.jsonl。',
                file=sys.stderr,
            )
            print('当前得到 %d 个路径参数：%s' % (len(paths), paths), file=sys.stderr)
            sys.exit(2)
        left_arg = ns.input
        li, right_arg, ri = paths
    else:
        if len(paths) != 4:
            print(
                '错误：需提供恰好 4 个路径：左图像目录、左 intrinsics.jsonl、右图像目录、右 intrinsics.jsonl。',
                file=sys.stderr,
            )
            print('或使用 --input <左目录> 后只跟上述后三项。当前参数：%s' % paths, file=sys.stderr)
            sys.exit(2)
        left_arg, li, right_arg, ri = paths

    left_files = expand_image_inputs([left_arg])
    right_files = expand_image_inputs([right_arg])
    if not left_files or not right_files:
        print('错误: 左右图像列表为空，请检查路径。', file=sys.stderr)
        sys.exit(1)

    left_files = sorted(left_files)
    right_files = sorted(right_files)

    k0, d0 = load_intrinsics_jsonl(li)
    k1, d1 = load_intrinsics_jsonl(ri)

    pattern_wh = (pw, ph)
    ret, R, T, E, F, vlp, vrp, cl, cr, objp, img_sz, num_pairs = stereo_calibrate_from_images(
        left_files,
        right_files,
        k0,
        d0,
        k1,
        d1,
        pattern_wh,
        square_size,
    )

    out_dir = os.path.normpath(out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    def _ensure_parent(path):
        p = os.path.normpath(path)
        parent = os.path.dirname(os.path.abspath(p))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        return p

    extrinsics_path = ns.extrinsics_jsonl or os.path.join(out_dir, 'stereo_extrinsics.jsonl')
    meta_path = ns.stereo_meta_jsonl or os.path.join(out_dir, 'stereo_calibration_meta.jsonl')
    projection_path = ns.projection_json or os.path.join(out_dir, 'stereo_projection.json')
    extrinsics_path = _ensure_parent(extrinsics_path)
    meta_path = _ensure_parent(meta_path)
    projection_path = _ensure_parent(projection_path)

    extrinsics_only = {
        'R': R.astype(np.float64).tolist(),
        'T': T.reshape(-1).astype(np.float64).tolist(),
    }
    with open(extrinsics_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(extrinsics_only, ensure_ascii=False) + '\n')

    meta_record = {
        'rms': float(ret),
        'E': E.astype(np.float64).tolist(),
        'F': F.astype(np.float64).tolist(),
        'image_size': {'width': int(img_sz[0]), 'height': int(img_sz[1])},
        'pattern': {'inner_corners_width': pw, 'inner_corners_height': ph, 'square_size': square_size},
        'calibration_pairs': int(num_pairs),
        'left_images_count': len(left_files),
        'right_images_count': len(right_files),
        'left_intrinsics_file': os.path.normpath(li),
        'right_intrinsics_file': os.path.normpath(ri),
        'output_dir': out_dir,
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(meta_record, ensure_ascii=False) + '\n')

    P_rt = extrinsic_projection_matrix_RT(R, T)
    dist_left = d0.reshape(-1).astype(np.float64).tolist()
    dist_right = d1.reshape(-1).astype(np.float64).tolist()
    projection_bundle = {
        'convention': (
            'K_left、distortion_coefficients_left：左相机 intrinsics.jsonl 中的相机矩阵与畸变系数。'
            'K_right、distortion_coefficients_right：右相机。'
            'R（3×3）、T（3×1，此处存为长度 3 的向量）：stereoCalibrate 给出的右相机相对左相机的外参，X_right = R @ X + T。'
            'P（3×4）为外参组合矩阵 [R | T]，前三列为旋转，最后一列为平移（非 K@形式）。'
            'E：本质矩阵；F：基础矩阵；均由 stereoCalibrate 输出。'
        ),
        'K_left': k0.astype(np.float64).tolist(),
        'distortion_coefficients_left': dist_left,
        'K_right': k1.astype(np.float64).tolist(),
        'distortion_coefficients_right': dist_right,
        'R': R.astype(np.float64).tolist(),
        'T': T.reshape(-1).astype(np.float64).tolist(),
        'P': P_rt.astype(np.float64).tolist(),
        'E': E.astype(np.float64).tolist(),
        'F': F.astype(np.float64).tolist(),
    }
    write_projection_bundle_json_horizontal(projection_path, projection_bundle)

    print('Stereo RMS:', ret)
    print('R (right <- left rotation):\n', R)
    print('T (right <- left translation):\n', T.ravel())
    print('Relative extrinsics (R, T only) saved to:', extrinsics_path)
    print('Stereo calibration meta saved to:', meta_path)
    print('Stereo summary (intrinsics, R/T, P=[R|T], E, F) saved to:', projection_path)

    # 坐标轴检查图
    if vlp and vrp and cl is not None and cr is not None:
        im0 = cv.imread(vlp)
        im1 = cv.imread(vrp)
        if im0 is not None and im1 is not None:
            vis0, vis1 = draw_axes_pair(
                im0,
                im1,
                k0,
                d0,
                k1,
                d1,
                cl,
                objp,
                R,
                T,
                axis_scale,
            )
            p0 = os.path.join(out_dir, 'axes_check_left.png')
            p1 = os.path.join(out_dir, 'axes_check_right.png')
            cv.imwrite(p0, vis0)
            cv.imwrite(p1, vis1)
            print('Axes overlay saved:', p0, p1)
            try:
                cv.imshow('axes_left', vis0)
                cv.imshow('axes_right', vis1)
                print('按任意键关闭坐标轴预览窗口...')
                cv.waitKey(0)
                cv.destroyAllWindows()
            except Exception:
                cv.destroyAllWindows()


if __name__ == '__main__':
    main()

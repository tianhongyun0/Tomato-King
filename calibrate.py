#!/usr/bin/env python

'''
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images

usage:
    calibrate.py [-i <dir|通配符>] [--input <dir|通配符>] [--output <dir>] [--debug <dir>]
    [-w <width>] [-h <height>] [-t <pattern type>] [--square_size=<square size>] ...

usage example:
    python calibrate.py -w 9 -h 6 -t chessboard --square_size=10 -o ./output/left/ -i ./data/left/
    calibrate.py -w 4 -h 6 -t chessboard --square_size=50 --input ./data/
    calibrate.py ... "./data/*.jpg"    （仍可用位置参数，与 -i 可同时使用，前后拼接）

default values:
    --output / --debug:  ./output   （结果图、调试图写入此目录；可用 -o / --output 覆盖）
    输入图像:              ./data    （未指定路径时扫描该文件夹内常见图像后缀，不含子目录）
    -w: 4
    -h: 6
    -t: chessboard
    --square_size: 10
    --marker_size: 5
    --aruco_dict: DICT_4X4_50
    --threads: 4

    输出：在 --output 目录下写入 intrinsics.jsonl（单行 JSON，供 calib.py 读取）、各图 *_board.png、*_undistorted.png。

NOTE: Chessboard size is defined in inner corners. Charuco board size is defined in units.
'''

# Python 2/3 compatibility,为了兼容python2和python3，print函数在python3中是一个函数，在python2中是一个语句，所以需要导入print_function来使用python3的print函数。
from __future__ import print_function

import numpy as np# 主要处理组和矩阵的运算，主要用来生成3维的坐标
import cv2 as cv

# local modules
from common import splitfn#在本地的common.py文件中定义了一个splitfn函数，用于分割文件路径、文件名和扩展名。
# def splitfn(fn):
#     path,fn = os.path.split(fn)
#     name,ext= os.splitext(fn)
#     return path,name,ext

# built-in modules
import json
import os  # 创建目录、处理文件夹路径、判断文件是否存在
from glob import glob

#读取多种类型的图像，定义了一个包含常见图像扩展名的元组IMAGE_EXTS，以及一个函数expand_image_inputs()，用于将命令行传入的目录或通配符展开为图像文件列表。
IMAGE_EXTS = (
    '.jpg', '.jpeg',
    '.png',
    '.bmp', '.dib',
    '.tif', '.tiff',
    '.webp',
    '.jp2',
    '.pbm', '.pgm', '.ppm', '.pnm',
    '.sr', '.ras',
    '.exr',
    '.hdr', '.pic',
    '.imp',   
)

def expand_image_inputs(inputs):
    """将命令行传入的目录/通配符展开为图像文件列表；无参数时默认使用 ./data。"""
    files = []

    # 未指定输入路径时，默认读取项目下的 ./data 目录
    if not inputs:
        inputs = [os.path.normpath('./data')]

    for item in inputs:
        item = os.path.normpath(item)
        # 情况1：传入的是文件夹，例如 ./data/left/
        if os.path.isdir(item):
            for ext in IMAGE_EXTS:
                files.extend(glob(os.path.join(item, '*' + ext)))
                files.extend(glob(os.path.join(item, '*' + ext.upper())))

        # 情况2：传入的是通配符，例如 ./data/left/*.png
        else:
            hits = glob(item)
            if hits:
                files.extend([x for x in hits if os.path.isfile(x)])
            elif os.path.isfile(item):
                files.append(item)

    # 去重 + 排序
    files = sorted(set(files))

    return files


def main():
    import sys
    import getopt

    args, img_names = getopt.getopt(sys.argv[1:], 'w:h:t:o:i:', ['debug=', 'output=', 'input=', 'square_size=', 'marker_size=',
                                                                'aruco_dict=', 'threads=', ])
    args = dict(args)
    args.setdefault('-w', 4)
    args.setdefault('-h', 6)
    args.setdefault('-t', 'chessboard')
    args.setdefault('--square_size', 10)
    args.setdefault('--marker_size', 5)
    args.setdefault('--aruco_dict', 'DICT_4X4_50')
    args.setdefault('--threads', 4)

    # 输入路径：优先 -i / --input（无需给路径加引号）；可与位置参数叠加
    input_paths = []
    if args.get('-i'):
        input_paths.append(args['-i'])
    if args.get('--input'):
        input_paths.append(args['--input'])
    if input_paths:
        img_names = input_paths + list(img_names)

    img_names = expand_image_inputs(img_names)

    if not img_names:
        print('错误: 没有匹配到任何图像，请检查路径或通配符。', file=sys.stderr)
        return

    # 输出目录：默认 ./output；可用 --output / -o / --debug（后者兼容旧用法）
    debug_dir = args.get('--output') or args.get('-o') or args.get('--debug') or './output'
    debug_dir = os.path.normpath(debug_dir)
    if debug_dir and not os.path.isdir(debug_dir):
        os.makedirs(debug_dir)

    height = int(args.get('-h'))
    width = int(args.get('-w'))
    pattern_type = str(args.get('-t'))
    square_size = float(args.get('--square_size'))
    marker_size = float(args.get('--marker_size'))
    aruco_dict_name = str(args.get('--aruco_dict'))

#转化为真实的棋盘格尺寸
    pattern_size = (width, height)
    pattern_points = None
    if pattern_type == 'chessboard':
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)#np.prod代表乘积，现在表示24个点，每个点有3个坐标值，初始值为0
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)#np.indices(pattern_size)生成一个包含棋盘格坐标的数组，shape为(2, height, width)，其中第一个维度表示x和y坐标，第二个和第三个维度表示棋盘格的行和列。通过.T转置和reshape(-1, 2)将其转换为一个形状为(height*width, 2)的二维数组，每行表示一个棋盘格点的(x, y)坐标。
        pattern_points *= square_size

#物体坐标系的空列表，像素点坐标
    obj_points = []
    img_points = []
#读取图像的尺寸，读取第一张图片，用灰度的读取方式，[:2]代表读取shape的前两个值，即高度和宽度。
    first = cv.imread(img_names[0], cv.IMREAD_GRAYSCALE)
    if first is None:
        print('错误: 无法读取图像:', img_names[0], file=sys.stderr)
        return
    h, w = first.shape[:2]  # TODO: use imquery call to retrieve results

    board = None
    charuco_detector = None
    if pattern_type == 'charucoboard':
        aruco_dicts = {
            'DICT_4X4_50': cv.aruco.DICT_4X4_50,
            'DICT_4X4_100': cv.aruco.DICT_4X4_100,
            'DICT_4X4_250': cv.aruco.DICT_4X4_250,
            'DICT_4X4_1000': cv.aruco.DICT_4X4_1000,
            'DICT_5X5_50': cv.aruco.DICT_5X5_50,
            'DICT_5X5_100': cv.aruco.DICT_5X5_100,
            'DICT_5X5_250': cv.aruco.DICT_5X5_250,
            'DICT_5X5_1000': cv.aruco.DICT_5X5_1000,
            'DICT_6X6_50': cv.aruco.DICT_6X6_50,
            'DICT_6X6_100': cv.aruco.DICT_6X6_100,
            'DICT_6X6_250': cv.aruco.DICT_6X6_250,
            'DICT_6X6_1000': cv.aruco.DICT_6X6_1000,
            'DICT_7X7_50': cv.aruco.DICT_7X7_50,
            'DICT_7X7_100': cv.aruco.DICT_7X7_100,
            'DICT_7X7_250': cv.aruco.DICT_7X7_250,
            'DICT_7X7_1000': cv.aruco.DICT_7X7_1000,
            'DICT_ARUCO_ORIGINAL': cv.aruco.DICT_ARUCO_ORIGINAL,
            'DICT_APRILTAG_16h5': cv.aruco.DICT_APRILTAG_16h5,
            'DICT_APRILTAG_25h9': cv.aruco.DICT_APRILTAG_25h9,
            'DICT_APRILTAG_36h10': cv.aruco.DICT_APRILTAG_36h10,
            'DICT_APRILTAG_36h11': cv.aruco.DICT_APRILTAG_36h11
        }

        if aruco_dict_name not in aruco_dicts:
            print("unknown aruco dictionary name")
            return None
        if square_size <= marker_size:
            print(
                '错误: Charuco 板要求 square_size > marker_size（当前 square_size=%s, marker_size=%s）。'
                % (square_size, marker_size),
                file=sys.stderr,
            )
            return
        aruco_dict = cv.aruco.getPredefinedDictionary(aruco_dicts[aruco_dict_name])
        board = cv.aruco.CharucoBoard(pattern_size, square_size, marker_size, aruco_dict)
        charuco_detector = cv.aruco.CharucoDetector(board)

#处理图片
    def processImage(fn):
        print('processing %s... ' % fn)
        img = cv.imread(fn, cv.IMREAD_GRAYSCALE)
        if img is None:
            print("Failed to load", fn)
            return None

        assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
        found = False #是否找到角点
        corners = 0 #角点变量初始化
        if pattern_type == 'chessboard':
            found, corners = cv.findChessboardCorners(img, pattern_size) #输出是否找到角点和角点坐标
            if found:
                term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
                cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
                #变成一个2维的数组，每行表示一个角点的(x, y)坐标，reshape(-1, 2)表示去掉冗余维度，变成（N, 2）的2维数组
                frame_img_points = corners.reshape(-1, 2)
                frame_obj_points = pattern_points
        elif pattern_type == 'charucoboard':
            corners, charucoIds, _, _ = charuco_detector.detectBoard(img)
            if (len(corners) > 0):
                frame_obj_points, frame_img_points = board.matchImagePoints(corners, charucoIds)
                found = True
            else:
                found = False
        else:
            print("unknown pattern type", pattern_type)
            return None

        if debug_dir:
            vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR) #把灰度图转换为彩色图，以便在上面绘制角点
            if pattern_type == 'chessboard':
                cv.drawChessboardCorners(vis, pattern_size, corners, found)
            elif pattern_type == 'charucoboard':
                cv.aruco.drawDetectedCornersCharuco(vis, corners, charucoIds=charucoIds)
            _path, name, _ext = splitfn(fn)
            outfile = os.path.join(debug_dir, name + '_board.png')
            cv.imwrite(outfile, vis)

        if not found:
            print('pattern not found')
            return None

        print('           %s... OK' % fn)
        return (frame_img_points, frame_obj_points)

#多线程处理图片
    threads_num = int(args.get('--threads'))
    if threads_num <= 1:
        chessboards = [processImage(fn) for fn in img_names]
    else:
        print("Run with %d threads..." % threads_num)
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool(threads_num)
        chessboards = pool.map(processImage, img_names)

#过滤掉没有找到角点的图片，并把找到的角点坐标和对应的物体坐标添加到列表中，准备进行相机校准
    chessboards = [x for x in chessboards if x is not None]
    for (corners, pattern_points) in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)

    # calculate camera distortion计算相机参数，输出RMS误差、相机矩阵、畸变系数等参数，并使用这些参数对图像进行去畸变处理，最后保存去畸变后的图像。
    rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)

    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())

    # 与 calib.py 的 load_intrinsics_jsonl 约定一致：单行 JSON，含 camera_matrix、distortion_coefficients
    intrinsics_record = {
        "rms": float(rms),
        "image_width": int(w),
        "image_height": int(h),
        "pattern_columns": int(width),
        "pattern_rows": int(height),
        "square_size": float(square_size),
        "pattern_type": pattern_type,
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coefs.reshape(-1).tolist(),
    }
    intrinsics_path = os.path.join(debug_dir, "intrinsics.jsonl")
    with open(intrinsics_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(intrinsics_record, ensure_ascii=False) + "\n")
    print("内参已写入:", intrinsics_path)

    # undistort the image with the calibration
    print('')
    for fn in img_names if debug_dir else []:
        _path, name, _ext = splitfn(fn)
        img_found = os.path.join(debug_dir, name + '_board.png')
        outfile = os.path.join(debug_dir, name + '_undistorted.png')

        img = cv.imread(img_found)
        if img is None:
            continue

        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

        dst = cv.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

        # crop and save the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        print('Undistorted image written to: %s' % outfile)
        cv.imwrite(outfile, dst)

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
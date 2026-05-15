conda activate D:\conda_envs\Hand_DR
conda activate D:\conda_envs\yolov12

相机内参使用说明
-
python calibrate.py -w 11 -h 7 -t chessboard --square_size=2.5 -o ./output/left/ -i ./data/left/
-w 11:棋盘格横向角点数。-h 7：棋盘格纵向角点数。-t chessboard:使用棋盘格。--square_size=2.5:棋盘格的实际尺寸，单位是cm。 ../data/left 读取哪些图片进行标定。 -o ./output/left 输出路径。

相机外参使用说明（路径顺序：左图目录 → 左 intrinsics.jsonl → 右图目录 → 右 intrinsics.jsonl）
python calib.py -w 11 -h 7 --square_size 2.5 ./data/left/ ./output/left/intrinsics.jsonl ./data/right/ ./output/right/intrinsics.jsonl -o ./output/stereo
python calib.py -w 11 -h 7 --square_size 2.5 --input ./data/left/ ./output/left/intrinsics.jsonl ./data/right/ ./output/right/intrinsics.jsonl --output ./output/stereo/

convert_to_bmp.py  python convert_to_bmp.py input.png -o output.bmp

python laser_spot_center.py --input-dir input --out input\debug_result --mode bright  

python points_3d.py input\single_spot_manifest.csv -o input\single_spot_manifest1.csv

python estimate_extrinsic_pnp.py --points_csv input\single_spot_manifest1.csv --centers_json input\debug_result\center_result.jsonl --intrinsics_json output\left1\intrinsics.jsonl --out_dir output\extrinsic_result --holdout loo  --remove_worst --remove_by loo

python fit_sphere_from_total_station.py --input input\single_spot_manifest.csv --output output\sphere_params.json
python fit_sphere_from_total_station.py -i input\single_spot_manifest.csv -o output\sphere_params.json

python sphere_fitting.py -i input\single_spot_manifest.csv -o output\sphere_params.json --method all

python sphere_fitting3.py -i input\single_spot_manifest.csv -o output\sphere_params.json --method all

python sphere_surface_rbf.py -i input\single_spot_manifest.csv -o output\sphere_params.json --local-rbf-csv input\new_points.csv --rbf-radius-mm 500
python sphere_fit_freeform_surface.py -i input\single_spot_manifest.csv -o output\sphere_params.json --global-degree 2
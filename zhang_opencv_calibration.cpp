#define NOMINMAX

#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>
#include <filesystem>
#include <cmath>
#include <cwctype>
#include <set>

#ifdef _WIN32
#include <windows.h>
#endif

namespace zhang_opencv_calibration
{
    namespace fs = std::filesystem;

    struct ZhangConfig
    {
        // 原始标定图像文件夹
        std::wstring inputDir =
            L"G:\\PythonAlgorithm\\PythonAlgorithm\\博一\\experimentdata\\华科船舶实验室内参标定图像";

        // 输出结果文件夹
        std::wstring outputDir =
            L"G:\\PythonAlgorithm\\PythonAlgorithm\\博一\\experimentdata\\output\\华科船舶结果\\opencv原图结果";

        // 棋盘格内角点数量，不是方格数量
        // 例如内角点为 5 × 8，就写 5 和 8
        int boardW = 5;
        int boardH = 8;

        // 单个棋盘格方格实际尺寸，单位 mm
        double squareSize = 70.0;

        // 是否固定 k3，与你之前 OpenCV/Qt 标定保持一致
        bool fixK3 = true;

        // 是否保存角点可视化图
        bool saveCornerVisualization = true;

        // 是否保存每张图像编号
        bool drawCornerIndex = true;

        // 角点亚像素窗口
        cv::Size subpixWinSize = cv::Size(11, 11);

        // OpenCV findChessboardCorners 检测参数
        int chessFlags =
            cv::CALIB_CB_ADAPTIVE_THRESH |
            cv::CALIB_CB_NORMALIZE_IMAGE |
            cv::CALIB_CB_FILTER_QUADS;
    };

    static std::string wstringToUtf8(const std::wstring& wstr)
    {
#ifdef _WIN32
        if (wstr.empty()) {
            return std::string();
        }

        int sizeNeeded = WideCharToMultiByte(
            CP_UTF8,
            0,
            wstr.c_str(),
            static_cast<int>(wstr.size()),
            nullptr,
            0,
            nullptr,
            nullptr
        );

        std::string str(sizeNeeded, 0);

        WideCharToMultiByte(
            CP_UTF8,
            0,
            wstr.c_str(),
            static_cast<int>(wstr.size()),
            str.data(),
            sizeNeeded,
            nullptr,
            nullptr
        );

        return str;
#else
        return std::string(wstr.begin(), wstr.end());
#endif
    }

    static cv::Mat imreadUnicode(const fs::path& path, int flags = cv::IMREAD_COLOR)
    {
        std::ifstream file(path, std::ios::binary);

        if (!file.is_open()) {
            return cv::Mat();
        }

        std::vector<uchar> buffer(
            (std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>()
        );

        if (buffer.empty()) {
            return cv::Mat();
        }

        return cv::imdecode(buffer, flags);
    }

    static bool imwriteUnicode(const fs::path& path, const cv::Mat& image)
    {
        std::string ext = path.extension().string();

        if (ext.empty()) {
            ext = ".png";
        }

        std::vector<uchar> buffer;

        if (!cv::imencode(ext, image, buffer)) {
            return false;
        }

        std::ofstream file(path, std::ios::binary);

        if (!file.is_open()) {
            return false;
        }

        file.write(
            reinterpret_cast<const char*>(buffer.data()),
            static_cast<std::streamsize>(buffer.size())
        );

        return true;
    }

    static bool isImageFile(const fs::path& path)
    {
        std::wstring ext = path.extension().wstring();

        std::transform(ext.begin(), ext.end(), ext.begin(), [](wchar_t c) {
            return static_cast<wchar_t>(std::towlower(c));
            });

        return ext == L".jpg" ||
            ext == L".jpeg" ||
            ext == L".png" ||
            ext == L".bmp" ||
            ext == L".tif" ||
            ext == L".tiff";
    }

    static std::vector<fs::path> collectImages(const fs::path& inputDir)
    {
        std::vector<fs::path> imagePaths;

        if (!fs::exists(inputDir)) {
            std::wcerr << L"[ERROR] Input directory does not exist: "
                << inputDir.wstring() << std::endl;
            return imagePaths;
        }

        for (const auto& entry : fs::directory_iterator(inputDir)) {
            if (entry.is_regular_file() && isImageFile(entry.path())) {
                imagePaths.push_back(entry.path());
            }
        }

        std::sort(imagePaths.begin(), imagePaths.end());

        return imagePaths;
    }

    static cv::Mat toGray(const cv::Mat& src)
    {
        cv::Mat gray;

        if (src.empty()) {
            return gray;
        }

        if (src.channels() == 1) {
            gray = src.clone();
        }
        else if (src.channels() == 3) {
            cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        }
        else if (src.channels() == 4) {
            cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
        }
        else {
            gray = src.clone();
        }

        return gray;
    }

    static std::vector<cv::Point3f> buildObjectPoints(
        int boardW,
        int boardH,
        double squareSize
    )
    {
        std::vector<cv::Point3f> objectPoints;

        for (int y = 0; y < boardH; ++y) {
            for (int x = 0; x < boardW; ++x) {
                objectPoints.emplace_back(
                    static_cast<float>(x * squareSize),
                    static_cast<float>(y * squareSize),
                    0.0f
                );
            }
        }

        return objectPoints;
    }

    static double getDistCoeff(const cv::Mat& distCoeffs, int idx)
    {
        if (idx < 0 || idx >= static_cast<int>(distCoeffs.total())) {
            return 0.0;
        }

        return distCoeffs.ptr<double>()[idx];
    }

    static std::vector<double> computePerViewErrors(
        const std::vector<std::vector<cv::Point3f>>& objectPoints,
        const std::vector<std::vector<cv::Point2f>>& imagePoints,
        const std::vector<cv::Mat>& rvecs,
        const std::vector<cv::Mat>& tvecs,
        const cv::Mat& cameraMatrix,
        const cv::Mat& distCoeffs
    )
    {
        std::vector<double> perViewErrors;

        for (size_t i = 0; i < objectPoints.size(); ++i) {
            std::vector<cv::Point2f> projectedPoints;

            cv::projectPoints(
                objectPoints[i],
                rvecs[i],
                tvecs[i],
                cameraMatrix,
                distCoeffs,
                projectedPoints
            );

            double err = cv::norm(
                imagePoints[i],
                projectedPoints,
                cv::NORM_L2
            );

            double rms = std::sqrt(
                err * err / static_cast<double>(projectedPoints.size())
            );

            perViewErrors.push_back(rms);
        }

        return perViewErrors;
    }

    static bool saveCamFile(
        const fs::path& filePath,
        const cv::Mat& cameraMatrix,
        const cv::Mat& distCoeffs
    )
    {
        std::ofstream out(filePath);

        if (!out.is_open()) {
            return false;
        }

        out << std::fixed << std::setprecision(12);

        out << "Camera Name = OpenCV_Zhang_Calibration\n";
        out << "Calibrated = 1\n";

        out << "fx = " << cameraMatrix.at<double>(0, 0) << "\n";
        out << "fy = " << cameraMatrix.at<double>(1, 1) << "\n";
        out << "cx = " << cameraMatrix.at<double>(0, 2) << "\n";
        out << "cy = " << cameraMatrix.at<double>(1, 2) << "\n";

        out << "k1 = " << getDistCoeff(distCoeffs, 0) << "\n";
        out << "k2 = " << getDistCoeff(distCoeffs, 1) << "\n";
        out << "p1 = " << getDistCoeff(distCoeffs, 2) << "\n";
        out << "p2 = " << getDistCoeff(distCoeffs, 3) << "\n";
        out << "k3 = " << getDistCoeff(distCoeffs, 4) << "\n";

        out.close();

        return true;
    }

    static bool saveReport(
        const fs::path& filePath,
        const ZhangConfig& cfg,
        double rms,
        const cv::Size& imageSize,
        const cv::Mat& cameraMatrix,
        const cv::Mat& distCoeffs,
        const std::vector<double>& perViewErrors,
        const std::vector<fs::path>& validImagePaths
    )
    {
        std::ofstream fout(filePath, std::ios::binary);

        if (!fout.is_open()) {
            return false;
        }

        // UTF-8 BOM，方便 Windows 记事本打开
        unsigned char bom[] = { 0xEF, 0xBB, 0xBF };
        fout.write(reinterpret_cast<char*>(bom), 3);

        fout << std::fixed << std::setprecision(12);

        fout << "OpenCV Zhang Camera Calibration Report\n";
        fout << "=====================================\n\n";

        fout << "This program uses OpenCV findChessboardCornersSB  + calibrateCamera.\n";
        fout << "The calibration board is treated as a planar checkerboard target with Z = 0.\n\n";

        fout << "Input image size: "
            << imageSize.width << " x " << imageSize.height << "\n";

        fout << "Board inner corners: "
            << cfg.boardW << " x " << cfg.boardH << "\n";

        fout << "Square size: "
            << cfg.squareSize << " mm\n";

        fout << "Valid calibration images: "
            << validImagePaths.size() << "\n";

        fout << "Calibration RMS reprojection error: "
            << rms << " pixels\n\n";

        fout << "Calibration flags:\n";

        if (cfg.fixK3) {
            fout << "cv::CALIB_FIX_K3 enabled\n";
        }
        else {
            fout << "k3 estimated\n";
        }

        fout << "\nCamera Matrix:\n";

        for (int i = 0; i < 3; ++i) {
            fout << cameraMatrix.at<double>(i, 0) << " "
                << cameraMatrix.at<double>(i, 1) << " "
                << cameraMatrix.at<double>(i, 2) << "\n";
        }

        fout << "\nDistortion Coefficients:\n";

        fout << "k1 = " << getDistCoeff(distCoeffs, 0) << "\n";
        fout << "k2 = " << getDistCoeff(distCoeffs, 1) << "\n";
        fout << "p1 = " << getDistCoeff(distCoeffs, 2) << "\n";
        fout << "p2 = " << getDistCoeff(distCoeffs, 3) << "\n";
        fout << "k3 = " << getDistCoeff(distCoeffs, 4) << "\n";

        fout << "\nPer-image reprojection RMS error:\n";

        for (size_t i = 0; i < perViewErrors.size(); ++i) {
            fout << "Image " << i + 1 << ": "
                << perViewErrors[i] << " pixels    ";

            if (i < validImagePaths.size()) {
                fout << wstringToUtf8(validImagePaths[i].filename().wstring());
            }

            fout << "\n";
        }

        fout.close();

        return true;
    }

    static bool saveOpenCVYaml(
        const fs::path& filePath,
        const cv::Mat& cameraMatrix,
        const cv::Mat& distCoeffs,
        const cv::Size& imageSize,
        double rms
    )
    {
        std::string pathUtf8 = wstringToUtf8(filePath.wstring());

        cv::FileStorage fsout(pathUtf8, cv::FileStorage::WRITE);

        if (!fsout.isOpened()) {
            return false;
        }

        fsout << "image_width" << imageSize.width;
        fsout << "image_height" << imageSize.height;
        fsout << "camera_matrix" << cameraMatrix;
        fsout << "distortion_coefficients" << distCoeffs;
        fsout << "rms_reprojection_error" << rms;

        fsout.release();

        return true;
    }

    static int run()
    {
#ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
#endif

        ZhangConfig cfg;

        fs::path inputDir(cfg.inputDir);
        fs::path outputDir(cfg.outputDir);
        fs::path visDir = outputDir / L"corner_visualization";

        fs::create_directories(outputDir);

        if (cfg.saveCornerVisualization) {
            fs::create_directories(visDir);
        }

        std::cout << "========================================\n";
        std::cout << "OpenCV Zhang Camera Calibration\n";
        std::cout << "findChessboardCorners + cornerSubPix + calibrateCamera\n";
        std::cout << "========================================\n\n";

        std::wcout << L"[INFO] Input directory: "
            << inputDir.wstring() << std::endl;

        std::wcout << L"[INFO] Output directory: "
            << outputDir.wstring() << std::endl;

        std::cout << "[INFO] Board inner corners: "
            << cfg.boardW << " x " << cfg.boardH << std::endl;

        std::cout << "[INFO] Square size: "
            << cfg.squareSize << " mm\n\n";

        std::vector<fs::path> imagePaths = collectImages(inputDir);

        if (imagePaths.empty()) {
            std::cerr << "[ERROR] No images found.\n";
            system("pause");
            return -1;
        }

        std::cout << "[INFO] Found images: "
            << imagePaths.size() << "\n\n";

        cv::Mat firstImage = imreadUnicode(imagePaths[0], cv::IMREAD_COLOR);

        if (firstImage.empty()) {
            std::cerr << "[ERROR] Failed to read first image.\n";
            system("pause");
            return -1;
        }

        cv::Size imageSize = firstImage.size();

        std::cout << "[INFO] Image size: "
            << imageSize.width << " x "
            << imageSize.height << "\n\n";

        cv::Size patternSize(cfg.boardW, cfg.boardH);

        std::vector<std::vector<cv::Point2f>> imagePointsAll;
        std::vector<std::vector<cv::Point3f>> objectPointsAll;
        std::vector<fs::path> validImagePaths;

        std::vector<cv::Point3f> singleObjectPoints =
            buildObjectPoints(cfg.boardW, cfg.boardH, cfg.squareSize);

        for (size_t i = 0; i < imagePaths.size(); ++i) {
            const fs::path& imgPath = imagePaths[i];

            std::wcout << L"[PROCESS] "
                << i + 1 << L" / "
                << imagePaths.size() << L": "
                << imgPath.filename().wstring()
                << std::endl;

            cv::Mat image = imreadUnicode(imgPath, cv::IMREAD_COLOR);

            if (image.empty()) {
                std::cout << "  [SKIP] Failed to read image.\n";
                continue;
            }

            if (image.size() != imageSize) {
                std::cout << "  [SKIP] Image size is different from first image.\n";
                continue;
            }

            cv::Mat gray = toGray(image);

            if (gray.empty()) {
                std::cout << "  [SKIP] Failed to convert to gray.\n";
                continue;
            }

            std::vector<cv::Point2f> corners;

            // findChessboardCornersSB 的 flags 和传统 findChessboardCorners 不完全一样
            int sbFlags =
                cv::CALIB_CB_NORMALIZE_IMAGE |
                cv::CALIB_CB_EXHAUSTIVE |
                cv::CALIB_CB_ACCURACY;

            bool found = cv::findChessboardCornersSB(
                gray,
                patternSize,
                corners,
                sbFlags
            );

            if (!found) {
                std::cout << "  [FAIL] findChessboardCornersSB failed.\n";
                continue;
            }

            if (static_cast<int>(corners.size()) != cfg.boardW * cfg.boardH) {
                std::cout << "  [FAIL] Corner count mismatch. Expected "
                    << cfg.boardW * cfg.boardH
                    << ", got "
                    << corners.size()
                    << "\n";
                continue;
            }

            imagePointsAll.push_back(corners);
            objectPointsAll.push_back(singleObjectPoints);
            validImagePaths.push_back(imgPath);

            std::cout << "  [OK] Corners detected: "
                << corners.size() << "\n";

            if (cfg.saveCornerVisualization) {
                cv::Mat vis = image.clone();

                cv::drawChessboardCorners(
                    vis,
                    patternSize,
                    corners,
                    true
                );

                if (cfg.drawCornerIndex) {
                    for (size_t k = 0; k < corners.size(); ++k) {
                        cv::putText(
                            vis,
                            std::to_string(k),
                            corners[k] + cv::Point2f(5.0f, -5.0f),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.45,
                            cv::Scalar(0, 0, 255),
                            1
                        );
                    }
                }

                fs::path outPath =
                    visDir / (imgPath.stem().wstring() + L"_zhang_corners.png");

                imwriteUnicode(outPath, vis);
            }
        }

        std::cout << "\n[INFO] Valid calibration images: "
            << imagePointsAll.size() << "\n";

        if (imagePointsAll.empty()) {
            std::cerr << "[ERROR] No valid chessboard corners were found.\n";
            system("pause");
            return -1;
        }

        if (imagePointsAll.size() < 5) {
            std::cout << "[WARNING] Valid images are fewer than 5. Calibration may be unstable.\n";
        }

        cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat distCoeffs = cv::Mat::zeros(1, 5, CV_64F);

        std::vector<cv::Mat> rvecs;
        std::vector<cv::Mat> tvecs;

        int calibFlags = 0;

        if (cfg.fixK3) {
            calibFlags |= cv::CALIB_FIX_K3;
        }

        std::cout << "\n[INFO] Starting OpenCV calibrateCamera...\n";

        double rms = cv::calibrateCamera(
            objectPointsAll,
            imagePointsAll,
            imageSize,
            cameraMatrix,
            distCoeffs,
            rvecs,
            tvecs,
            calibFlags
        );

        std::vector<double> perViewErrors = computePerViewErrors(
            objectPointsAll,
            imagePointsAll,
            rvecs,
            tvecs,
            cameraMatrix,
            distCoeffs
        );

        std::cout << "\n[DONE] Zhang calibration complete.\n";
        std::cout << "[RESULT] Overall RMS reprojection error: "
            << rms << " pixels\n\n";

        std::cout << "Camera Matrix:\n"
            << cameraMatrix << "\n\n";

        std::cout << "Distortion Coefficients:\n"
            << distCoeffs << "\n\n";

        std::cout << "Per-image RMS:\n";

        for (size_t i = 0; i < perViewErrors.size(); ++i) {
            std::cout << "  [" << i + 1 << "] "
                << perViewErrors[i]
                << " pixels    ";
                std::wcout << validImagePaths[i].filename().wstring();
                std::cout << "\n";
        }

        fs::path camPath =
            outputDir / L"opencv_zhang_camera_parameters.cam";

        fs::path reportPath =
            outputDir / L"opencv_zhang_calibration_report.txt";

        fs::path yamlPath =
            outputDir / L"opencv_zhang_camera_parameters.yml";

        bool camSaved = saveCamFile(
            camPath,
            cameraMatrix,
            distCoeffs
        );

        bool reportSaved = saveReport(
            reportPath,
            cfg,
            rms,
            imageSize,
            cameraMatrix,
            distCoeffs,
            perViewErrors,
            validImagePaths
        );

        bool yamlSaved = saveOpenCVYaml(
            yamlPath,
            cameraMatrix,
            distCoeffs,
            imageSize,
            rms
        );

        if (camSaved) {
            std::wcout << L"\n[SAVED] Camera parameters .cam: "
                << camPath.wstring()
                << std::endl;
        }
        else {
            std::cout << "\n[ERROR] Failed to save .cam file.\n";
        }

        if (reportSaved) {
            std::wcout << L"[SAVED] Calibration report: "
                << reportPath.wstring()
                << std::endl;
        }
        else {
            std::cout << "[ERROR] Failed to save report.\n";
        }

        if (yamlSaved) {
            std::wcout << L"[SAVED] OpenCV YAML file: "
                << yamlPath.wstring()
                << std::endl;
        }
        else {
            std::cout << "[WARNING] Failed to save YAML file. This does not affect calibration result.\n";
        }

        if (cfg.saveCornerVisualization) {
            std::wcout << L"[SAVED] Corner visualization folder: "
                << visDir.wstring()
                << std::endl;
        }

        std::cout << "\nAll done.\n";
        system("pause");

        return 0;
    }
}

int runZhangOpenCVCalibrationApp()
{
    return zhang_opencv_calibration::run();
}
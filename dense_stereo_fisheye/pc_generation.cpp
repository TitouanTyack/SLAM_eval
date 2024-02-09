#include <filesystem>
#include <fstream>

#include "denseStereo.hpp"

#include <yaml-cpp/yaml.h>

std::vector<long long> extract_ts_from_path(std::string path) {

    std::string word;
    std::vector<std::string> row;
    std::vector<long long> ts_vec;

    for (const auto &entry : std::filesystem::directory_iterator(path)) {
        row.clear();

        std::stringstream str((std::string)entry.path());
        while (getline(str, word, '/'))
            row.push_back(word);

        std::size_t pos    = row.back().find(".");
        std::string ts_str = row.back().erase(pos, 4);
        ts_vec.push_back(std::stoll(ts_str));
    }

    // Sort vector
    std::sort(ts_vec.begin(), ts_vec.end());

    return ts_vec;
}

int main() {

    // Read param file
    YAML::Node config = YAML::LoadFile("param.yaml");

    // Load path and timestamps
    std::string path_camleft  = config["path cam left"].as<std::string>();
    std::string path_camright = config["path cam right"].as<std::string>();
    std::vector<long long> ts_left_vec  = extract_ts_from_path(path_camleft);
    std::vector<long long> ts_right_vec = extract_ts_from_path(path_camright);

    // Associate stereo pairs
    std::vector<std::pair<std::string, std::string>> left_right_pairs;
    for (auto ts_left : ts_left_vec) {
        double dt_min = config["dt_min"].as<double>() * 1e9;
        long long ts_right_assoc;

        for (auto ts_right : ts_right_vec) {
            double dt = std::abs(ts_right - ts_left);
            if (dt < dt_min) {
                dt_min         = dt;
                ts_right_assoc = ts_right;
            }
        }

        if (dt_min < config["dt_min"].as<double>() * 1e9)
            left_right_pairs.push_back({std::to_string(ts_left), std::to_string(ts_right_assoc)});
    }

    // Load parameters
    std::string file_name = "omni_parameters.yaml";
    denseStereo ds        = denseStereo(file_name);
    ds._vfov              = config["vfov"].as<double>();
    ds._ndisp             = config["ndisp"].as<double>();
    ds._wsize             = config["wsize"].as<double>();
    ds.InitRectifyMap();

    double dt_total = 0;

    for (auto left_right : left_right_pairs) {
        auto t0 = std::chrono::high_resolution_clock::now();

        cv::Mat img_left  = cv::imread(path_camleft + "/" + left_right.first + ".png", cv::IMREAD_COLOR);
        cv::Mat img_right = cv::imread(path_camright + "/" + left_right.second + ".png", cv::IMREAD_COLOR);

        // Downsample image
        cv::Mat small_left_img, small_right_img;
        cv::resize(img_left, small_left_img, cv::Size(), 0.5, 0.5);
        cv::resize(img_right, small_right_img, cv::Size(), 0.5, 0.5);

        cv::Mat rect_imgl, rect_imgr;
        cv::remap(small_left_img, rect_imgl, ds.smap[0][0], ds.smap[0][1], 1, 0);
        cv::remap(small_right_img, rect_imgr, ds.smap[1][0], ds.smap[1][1], 1, 0);

        // Disparity computation
        cv::Mat disp_img, depth_map;
        ds.DisparityImage(rect_imgl, rect_imgr, disp_img, depth_map);

        // Pointcloud computation
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud = ds.pcFromDepthMap(depth_map);

        // Timing
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = (double) std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        dt_total += dt;
        std::cout << "Timing : " << dt << " ms" << std::endl;

        // Writing pointcloud
        pcl_cloud->width                              = pcl_cloud->points.size();
        pcl_cloud->height                             = 1;
        pcl::io::savePCDFileASCII("cloud/" + left_right.first + ".pcd", *pcl_cloud);
    }

    std::cout << "Average timing : " << dt_total / (double) left_right_pairs.size() << " ms " << std::endl;

}
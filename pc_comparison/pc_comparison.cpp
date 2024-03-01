#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

void remove_negativex(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out) {

    cloud_out->clear();
    for (auto pt : cloud_in->points)
        if (pt.x > 0)
            cloud_out->points.push_back(pt);
}

void remove_aroundrobot(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out) {

    cloud_out->clear();
    for (auto pt : cloud_in->points)
        if (sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z) > 1)
            cloud_out->points.push_back(pt);
}

std::vector<long long> extract_ts_from_csv(std::string file_path) {

    std::vector<long long> ts_vec;
    std::vector<std::vector<std::string>> content;
    std::vector<std::string> row;
    std::string line, word;

    std::fstream file(file_path, ios::in);
    if (file.is_open()) {
        while (getline(file, line)) {
            row.clear();

            std::stringstream str(line);

            while (getline(str, word, ','))
                row.push_back(word);

            if (row.at(0) != "#timestamp [ns]")
                ts_vec.push_back(std::stoll(row.at(0)));
        }
    }

    return ts_vec;
}

std::vector<long long> extract_ts_from_path(std::string path) {

    std::string word;
    std::vector<std::string> row;
    std::vector<long long> ts_vec;

    for (const auto &entry : fs::directory_iterator(path)) {
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

pcl::visualization::PCLVisualizer::Ptr cloud_visu(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D viewer"));
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "sample cloud");
    viewer->setBackgroundColor(255, 255, 255);
    return viewer;
}

inline float avg_registration_err(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
                                  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out,
                                  double thresh) {

    // Search for a nearest neighbour for each points of the input cloud
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud_in);
    float avg_distance = 0;
    for (auto pt : cloud_out->points) {
        
        std::vector<int> pointIdxKNNSearch(1);
        std::vector<float> pointKNNSquaredDistance(1);

        kdtree.nearestKSearch(pt, 1, pointIdxKNNSearch, pointKNNSquaredDistance);
        if (pointKNNSquaredDistance.at(0) < thresh)
            avg_distance += pointKNNSquaredDistance.at(0);
    }
    avg_distance /= (float)cloud_out->size();

    return avg_distance;
}

int main() {

    // Read param file
    YAML::Node config = YAML::LoadFile("param.yaml");

    // Init pointcloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_raw(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_original(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_transformed(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out_raw(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);

    // Parse lidar scans
    std::string path_lidar              = config["path lidar cloud"].as<std::string>();
    std::vector<long long> ts_lidar_vec = extract_ts_from_path(path_lidar);

    // Parse vio cloud directory
    std::string path_vio              = config["path vio cloud"].as<std::string>();
    std::vector<long long> ts_vio_vec = extract_ts_from_path(path_vio);

    // Init transformation
    std::vector<float> data_T(16);
    data_T                 = config["T_init"].as<std::vector<float>>();
    Eigen::Matrix4f T_init = Eigen::Map<Eigen::Affine3f::MatrixType>(&data_T[0], 4, 4).transpose();

    // Associate scans
    std::vector<std::pair<std::string, std::string>> vio_lidar_pairs;
    for (auto ts_vio : ts_vio_vec) {
        double dt_min = config["dt_min"].as<double>() * 1e9;
        long long ts_lidar_assoc;

        for (auto ts_lidar : ts_lidar_vec) {
            double dt = std::abs(ts_lidar - ts_vio);
            if (dt < dt_min) {
                dt_min         = dt;
                ts_lidar_assoc = ts_lidar;
            }
        }

        if (dt_min < config["dt_min"].as<double>() * 1e9)
            vio_lidar_pairs.push_back({path_vio + "/" + std::to_string(ts_vio) + ".pcd",
                                       path_lidar + "/" + std::to_string(ts_lidar_assoc) + ".pcd"});
    }

    std::cout << "length : " << vio_lidar_pairs.size() << std::endl;

    float avg_score_accuracy     = 0;
    float avg_score_completeness = 0;
    int i_start                  = config["i_start"].as<int>();
    int i_stop                   = config["i_stop"].as<int>();
    int n_clouds                 = (i_stop - i_start);
    int n_cloud_to_eval          = config["n_cloud_to_eval"].as<int>();
    int ratio                    = std::round((double)n_clouds / (double)n_cloud_to_eval);

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    pcl::PointCloud<pcl::PointXYZ> Final;
    int counter = 0;
    for (int i = i_start; i < i_stop; i++) {
        if (i % ratio != 0)
            continue;

        if (pcl::io::loadPCDFile<pcl::PointXYZ>(vio_lidar_pairs.at(i).first, *cloud_in_raw) == -1) //* load the file
        {
            PCL_ERROR("Couldn't read file test_pcd.pcd \n");
            return (-1);
        }

        if (pcl::io::loadPCDFile<pcl::PointXYZ>(vio_lidar_pairs.at(i).second, *cloud_out_raw) == -1) //* load the file
        {
            PCL_ERROR("Couldn't read file test_pcd.pcd \n");
            return (-1);
        }

        // Remove negative x for a lighter pc
        remove_negativex(cloud_out_raw, cloud_out);
        remove_aroundrobot(cloud_in_raw, cloud_in);

        // Perform ICP
        pcl::transformPointCloud(*cloud_in, *cloud_in_original, T_init);
        icp.setInputSource(cloud_in_original);
        icp.setInputTarget(cloud_out);
        icp.setMaxCorrespondenceDistance(0.5);
        icp.align(Final);

        // Align VIO cloud
        pcl::transformPointCloud(*cloud_in_original, *cloud_in_transformed, icp.getFinalTransformation());

        std::cout << icp.getFinalTransformation() << std::endl;
        std::cout << icp.getFinalTransformation() * T_init << std::endl;

        // Update scores
        avg_score_accuracy += icp.getFitnessScore();
        avg_score_completeness += avg_registration_err(cloud_in_transformed, cloud_out, 0.5);
        counter++;
    }

    // Compute scores
    avg_score_accuracy /= (float)counter;
    avg_score_completeness /= (float)counter;

    std::cout << "Accuracy : " << avg_score_accuracy << std::endl;
    std::cout << "Completeness : " << avg_score_completeness << std::endl;


    // Color input cloud w.r.t. distance to output cloud
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_colored(new pcl::PointCloud<pcl::PointXYZRGB>);
    // for (uint32_t i = 0; i < icp.correspondences_->size(); i++) {
    //     pcl::Correspondence currentCorrespondence = (icp.correspondences_)->at(i);
    //     // std::cout << "Index of the source point: " << currentCorrespondence.index_query << std::endl;
    //     // std::cout << "Index of the matching target point: " << currentCorrespondence.index_match << std::endl;
    //     // std::cout << "Distance between the corresponding points: " << currentCorrespondence.distance << std::endl;
    //     // std::cout << "Weight of the confidence in the correspondence: " << currentCorrespondence.weight <<
    //     std::endl; pcl::PointXYZ pt_xyz = cloud_in->at(currentCorrespondence.index_query); pcl::PointXYZRGB
    //     pt_xyzrgb; pt_xyzrgb.x = pt_xyz.x; pt_xyzrgb.y = pt_xyz.y; pt_xyzrgb.z = pt_xyz.z;

    //     // Set color
    //     double color[3];
    //     pt_xyzrgb.r = (int)(255 * currentCorrespondence.distance / 0.1);
    //     pt_xyzrgb.g = 0;
    //     pt_xyzrgb.b = 0;
    //     pt_xyzrgb.a = 255;
    //     cloud_colored->push_back(pt_xyzrgb);
    // }


    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D viewer"));
    viewer->addPointCloud<pcl::PointXYZ>(cloud_in_transformed, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0, 0, "sample cloud");
    viewer->setBackgroundColor(255, 255, 255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_out, "LidarCloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1.0, 0, "LidarCloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "LidarCloud");
    viewer->spin();

    return (0);
}

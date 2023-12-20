#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>

namespace fs = std::filesystem;

std::vector<long long> extract_ts_from_csv(std::string file_path)
{

    std::vector<long long> ts_vec;
    std::vector<std::vector<std::string>> content;
    std::vector<std::string> row;
    std::string line, word;

    std::fstream file(file_path, ios::in);
    if (file.is_open())
    {
        while (getline(file, line))
        {
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

std::vector<long long> extract_ts_from_path(std::string path)
{

    std::string word;
    std::vector<std::string> row;
    std::vector<long long> ts_vec;

    for (const auto &entry : fs::directory_iterator(path))
    {
        row.clear();

        std::stringstream str((std::string)entry.path());
        while (getline(str, word, '/'))
            row.push_back(word);

        std::size_t pos = row.back().find(".");
        std::string ts_str = row.back().erase(pos, 4);
        ts_vec.push_back(std::stoll(ts_str));
    }

    return ts_vec;
}

pcl::visualization::PCLVisualizer::Ptr cloud_visu(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D viewer"));
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "sample cloud");
    viewer->setBackgroundColor(255,255,255);
    return viewer;
}

int main()
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_transformed(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);

    // Parse lidar scans
    std::string path_lidar = "fisheye3_raw/mav0/lidar0/data";
    std::vector<long long> ts_lidar_vec = extract_ts_from_path(path_lidar);

    // Parse vio cloud directory
    std::string path_vio = "fisheye3_viocloud";
    std::vector<long long> ts_vio_vec = extract_ts_from_path(path_vio);

    // Associate scans
    std::vector<std::pair<std::string, std::string>> vio_lidar_pairs;
    for (auto ts_vio : ts_vio_vec)
    {
        double dt_min = 100000000000000;
        long long ts_lidar_assoc;

        for (auto ts_lidar : ts_lidar_vec)
        {
            double dt = std::abs(ts_lidar - ts_vio);
            if (dt < dt_min)
            {
                dt_min = dt;
                ts_lidar_assoc = ts_lidar;
            }
        }

        vio_lidar_pairs.push_back({path_vio + "/" + std::to_string(ts_vio) + ".pcd",
                                   path_lidar + "/" + std::to_string(ts_lidar_assoc) + ".pcd"});
    }

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(vio_lidar_pairs.at(30).first, *cloud_in) == -1) //* load the file
    {
        PCL_ERROR("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(vio_lidar_pairs.at(30).second, *cloud_out) == -1) //* load the file
    {
        PCL_ERROR("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(cloud_in);
    icp.setInputTarget(cloud_out);
    icp.setMaxCorrespondenceDistance(0.5);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);

    // Align VIO cloud
    pcl::transformPointCloud(*cloud_in, *cloud_in_transformed, icp.getFinalTransformation());
    pcl::CorrespondencesPtr coresp = icp.correspondences_;

    // Color input cloud w.r.t. distance to output cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_colored(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (uint32_t i = 0; i < icp.correspondences_->size(); i++)
    {
        pcl::Correspondence currentCorrespondence = (icp.correspondences_)->at(i);
        // std::cout << "Index of the source point: " << currentCorrespondence.index_query << std::endl;
        // std::cout << "Index of the matching target point: " << currentCorrespondence.index_match << std::endl;
        // std::cout << "Distance between the corresponding points: " << currentCorrespondence.distance << std::endl;
        // std::cout << "Weight of the confidence in the correspondence: " << currentCorrespondence.weight << std::endl;
        pcl::PointXYZ pt_xyz = cloud_in_transformed->at(currentCorrespondence.index_query);
        pcl::PointXYZRGB pt_xyzrgb;
        pt_xyzrgb.x = pt_xyz.x;
        pt_xyzrgb.y = pt_xyz.y;
        pt_xyzrgb.z = pt_xyz.z;

        // Set color
        double color[3];
        pt_xyzrgb.r = (int)(255 * currentCorrespondence.distance / 0.1);
        pt_xyzrgb.g = 0;
        pt_xyzrgb.b = 0;
        pt_xyzrgb.a = 255;
        cloud_colored->push_back(pt_xyzrgb);
    }

    std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
    std::cout << icp.getFinalTransformation() << std::endl;

    pcl::visualization::PCLVisualizer::Ptr viewer = cloud_visu(cloud_colored);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_out, "LidarCloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0,1.0,0, "LidarCloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "LidarCloud");
    viewer->spin();

    return (0);
}
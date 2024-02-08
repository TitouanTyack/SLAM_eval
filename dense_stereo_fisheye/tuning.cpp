#include "denseStereo.hpp"

////////////////////////////////////////////////////////////////////////////////

denseStereo ds;
int vfov_bar = 0, width_bar = 0, height_bar = 0;
int vfov_max = 54, width_max = 300, height_max = 300;
int ndisp_bar = 1, wsize_bar = 2;
int ndisp_max = 6, wsize_max = 4;
bool changed = false;

////////////////////////////////////////////////////////////////////////////////

void OnTrackAngle(int, void *) {
    ds._vfov = 60 + vfov_bar;
    changed  = true;
}

////////////////////////////////////////////////////////////////////////////////

void OnTrackWidth(int, void *) {
    ds._width = ds._cap_cols - width_bar;
    if (ds._width % 2 == 1)
        ds._width--;
    changed = true;
}

////////////////////////////////////////////////////////////////////////////////

void OnTrackHeight(int, void *) {
    ds._height = ds._cap_rows - height_bar;
    if (ds._height % 2 == 1)
        ds._height--;
    changed = true;
}

////////////////////////////////////////////////////////////////////////////////

void OnTrackNdisp(int, void *) {
    ds._ndisp = 16 + 16 * ndisp_bar;
    changed  = true;
}

////////////////////////////////////////////////////////////////////////////////

void OnTrackWsize(int, void *) {
    ds._wsize = 3 + 2 * wsize_bar;
    changed   = true;
}

////////////////////////////////////////////////////////////////////////////////

pcl::visualization::PCLVisualizer::Ptr cloud_visu(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D viewer"));
    viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1.0, 0, "sample cloud");
    viewer->setBackgroundColor(255, 255, 255);
    viewer->spin();
    return viewer;
}

int main(int argc, char **argv) {
    std::string file_name = argc == 2 ? argv[1] : "omni_parameters.yaml";
    ds                    = denseStereo(file_name);
    ds.InitRectifyMap();

    cv::Mat left_img, right_img;
    left_img  = cv::imread("data/cnes_left.png", cv::IMREAD_COLOR);
    right_img = cv::imread("data/cnes_right.png", cv::IMREAD_COLOR);

    char win_name[256];
    sprintf(win_name, "Raw Image: %d x %d", ds._cap_cols, ds._cap_rows);
    std::string param_win_name(win_name);
    cv::namedWindow(param_win_name);

    cv::createTrackbar("V. FoV:  60    +", param_win_name, &vfov_bar, vfov_max, OnTrackAngle);
    cv::createTrackbar("Width:  1280 -", param_win_name, &width_bar, width_max, OnTrackWidth);
    cv::createTrackbar("Height: 1024 -", param_win_name, &height_bar, height_max, OnTrackHeight);

    std::string disp_win_name = "Disparity Image";
    cv::namedWindow(disp_win_name);
    cv::createTrackbar("Num Disp:  16 + 16 *", disp_win_name, &ndisp_bar, ndisp_max, OnTrackNdisp);
    cv::createTrackbar("Blk   Size :     3  +  2 *", disp_win_name, &wsize_bar, wsize_max, OnTrackWsize);

    cv::Mat raw_imgl, raw_imgr, rect_imgl, rect_imgr;
    while (1) {
        auto t0 = std::chrono::high_resolution_clock::now();
        if (changed) {
            ds.InitRectifyMap();
        }

        if (left_img.total() == 0) {
            std::cout << "Image capture error" << std::endl;
            exit(-1);
        }

        cv::Mat small_left_img, small_right_img;
        cv::resize(left_img, small_left_img, cv::Size(), 0.5, 0.5);
        cv::resize(right_img, small_right_img, cv::Size(), 0.5, 0.5);

        cv::remap(small_left_img, rect_imgl, ds.smap[0][0], ds.smap[0][1], 1, 0);
        cv::remap(small_right_img, rect_imgr, ds.smap[1][0], ds.smap[1][1], 1, 0);

        // Disparity computation
        cv::Mat disp_img, depth_map;
        ds.DisparityImage(rect_imgl, rect_imgr, disp_img, depth_map);

        // Pointcloud computation
        if (changed) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr _pcl_cloud = ds.pcFromDepthMap(depth_map);
            _pcl_cloud->width                              = _pcl_cloud->points.size();
            _pcl_cloud->height                             = 1;
            pcl::io::savePCDFileASCII("cloud.pcd", *_pcl_cloud);
        }

        // Display depth map
        cv::Mat depth_img;
        double minDepth, maxDepth;
        cv::minMaxLoc(depth_map, &minDepth, &maxDepth);
        cv::convertScaleAbs(depth_map, depth_img, 255 / (maxDepth - minDepth));

        if (changed) {
            auto t1 = std::chrono::high_resolution_clock::now();
            std::cout << "Timing : " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                      << std::endl;
            changed = false;
        }

        imshow("Left Image", small_left_img);
        imshow("Right Image", small_right_img);
        imshow("Rectified Left Image", rect_imgl);
        imshow("Rectified Right Image", rect_imgr);
        imshow("Disparity Image", disp_img);
        imshow("Depth Map", depth_img);

        /* // Apply Median Filtering to the depth map
         cv::Mat depth_filtered;
         cv::medianBlur(depth_map, depth_filtered, 5); // Adjust the kernel size as needed


         // Apply bilateral filtering to the depth map
         cv::Mat depth_filtered;
         cv::bilateralFilter(depth_map, depth_filtered, 5, 25, 25);

         // Display the filtered depth map
         cv::Mat depth_filtered_img;
         double minDepth, maxDepth;
         cv::minMaxLoc(depth_filtered, &minDepth, &maxDepth);
         cv::convertScaleAbs(depth_filtered, depth_filtered_img, 255 / (maxDepth - minDepth));
         cv::imshow("Filtered Depth Map", depth_filtered_img);   */

        // Enhance depth map visualization using colormap
        /*cv::Mat depth_colormap;
        double minDepth, maxDepth;
        cv::minMaxLoc(depth_map, &minDepth, &maxDepth);
        depth_map.convertTo(depth_map, CV_8U, 255.0 / (maxDepth - minDepth), -minDepth);
        cv::applyColorMap(depth_map, depth_colormap, cv::COLORMAP_JET);
        imshow("Depth Map", depth_colormap);*/

        char key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27)
            break;
    }

    return 0;
}

////////////////////////////////////////////////////////////////////////////////

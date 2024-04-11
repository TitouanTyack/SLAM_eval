#include "denseStereo.hpp"

inline double MatRowMul(cv::Mat m, double x, double y, double z, int r) {
    return m.at<double>(r, 0) * x + m.at<double>(r, 1) * y + m.at<double>(r, 2) * z;
}

denseStereo::denseStereo(std::string configfilepath) : _configfilepath(configfilepath) {

    // Load parameters
    cv::FileStorage fs(_configfilepath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cout << "Failed to open ini parameters" << std::endl;
        exit(-1);
    }

    cv::Size cap_size;
    fs["cam_model"] >> _cam_model;
    fs["cap_size"] >> cap_size;
    fs["Kl"] >> Kl;
    fs["xil"] >> xil;
    Rl = cv::Mat::eye(3, 3, CV_64F);

    fs["Rl"] >> Rl;
    fs["Kr"] >> Kr;

    fs["xir"] >> xir;
    fs["Rr"] >> Rr;
    fs["T"] >> Translation;

    if (_cam_model == "ds") {
        fs["alphal"] >> alphal;
        fs["alphar"] >> alphar;
    } else if (_cam_model == "omni") {
        fs["Dr"] >> Dr;
        fs["Dl"] >> Dl;
    }

    fs.release();

    _cap_cols = cap_size.width;
    _cap_rows = cap_size.height;
    _width    = _cap_cols;
    _height   = _cap_rows;
}

void denseStereo::InitUndistortRectifyMap(cv::Mat K,
                                          cv::Mat D,
                                          double xi,
                                          cv::Mat R,
                                          cv::Mat P,
                                          cv::Size size,
                                          cv::Mat &map1,
                                          cv::Mat &map2) {
    map1 = cv::Mat(size, CV_32F);
    map2 = cv::Mat(size, CV_32F);

    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    double s  = K.at<double>(0, 1);

    double xid = xi;

    double k1 = D.at<double>(0, 0);
    double k2 = D.at<double>(0, 1);
    double p1 = D.at<double>(0, 2);
    double p2 = D.at<double>(0, 3);

    cv::Mat KRi = (P * R).inv();

    for (int r = 0; r < size.height; ++r) {
        for (int c = 0; c < size.width; ++c) {
            double xc = MatRowMul(KRi, c, r, 1., 0);
            double yc = MatRowMul(KRi, c, r, 1., 1);
            double zc = MatRowMul(KRi, c, r, 1., 2);

            double rr = sqrt(xc * xc + yc * yc + zc * zc);
            double xs = xc / rr;
            double ys = yc / rr;
            double zs = zc / rr;

            double xu = xs / (zs + xid);
            double yu = ys / (zs + xid);

            double r2 = xu * xu + yu * yu;
            double r4 = r2 * r2;
            double xd = (1 + k1 * r2 + k2 * r4) * xu + 2 * p1 * xu * yu + p2 * (r2 + 2 * xu * xu);
            double yd = (1 + k1 * r2 + k2 * r4) * yu + 2 * p2 * xu * yu + p1 * (r2 + 2 * yu * yu);

            double u = fx * xd + s * yd + cx;
            double v = fy * yd + cy;

            map1.at<float>(r, c) = (float)u;
            map2.at<float>(r, c) = (float)v;
        }
    }
}

void denseStereo::InitUndistortRectifyMapDS(cv::Mat K,
                                            float alpha,
                                            float xi,
                                            cv::Mat R,
                                            cv::Mat P,
                                            cv::Size size,
                                            cv::Mat &map1,
                                            cv::Mat &map2) {
    map1 = cv::Mat(size, CV_32F);
    map2 = cv::Mat(size, CV_32F);

    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    double s  = K.at<double>(0, 1);

    cv::Mat KRi = (P * R).inv();

    for (int r = 0; r < size.height; ++r) {
        for (int c = 0; c < size.width; ++c) {
            double xc = MatRowMul(KRi, c, r, 1., 0);
            double yc = MatRowMul(KRi, c, r, 1., 1);
            double zc = MatRowMul(KRi, c, r, 1., 2);

            double rr = sqrt(xc * xc + yc * yc + zc * zc);
            double xs = xc / rr;
            double ys = yc / rr;
            double zs = zc / rr;

            double d1 = std::sqrt(xs * xs + ys * ys + zs * zs);
            double d2 = std::sqrt(xs * xs + ys * ys + (xi * d1 + zs) * (xi * d1 + zs));

            double xd = xs / (alpha * d2 + (1 - alpha) * (xi * d1 + zs));
            double yd = ys / (alpha * d2 + (1 - alpha) * (xi * d1 + zs));

            double u = fx * xd + s * yd + cx;
            double v = fy * yd + cy;

            map1.at<float>(r, c) = (float)u;
            map2.at<float>(r, c) = (float)v;
        }
    }
}

void denseStereo::InitRectifyMap() {

    double vfov_rad = _vfov * CV_PI / 180.;
    double focal    = _height / 2. / tan(vfov_rad / 2.);
    Knew = (cv::Mat_<double>(3, 3) << focal, 0., _width / 2. - 0.5, 0., focal, _height / 2. - 0.5, 0., 0., 1.);

    cv::Size img_size(_width, _height);
    if (_cam_model == "ds") {
        InitUndistortRectifyMapDS(
            Kl, alphal, xil, Rl, Knew, img_size, smap[0][0], smap[0][1]);
        InitUndistortRectifyMapDS(
            Kr, alphar, xir, Rr, Knew, img_size, smap[1][0], smap[1][1]);
    } else if (_cam_model == "omni") {
        InitUndistortRectifyMap(Kl, Dl, xil, Rl, Knew, img_size, smap[0][0], smap[0][1]);
        InitUndistortRectifyMap(Kr, Dr, xir, Rr, Knew, img_size, smap[1][0], smap[1][1]);
    }

    std::cout << "Width: " << _width << "\t"
              << "Height: " << _height << "\t"
              << "V.Fov: " << _vfov << "\n";
    std::cout << "K Matrix: \n" << Knew << std::endl;
    std::cout << "Ndisp: " << _ndisp << "\t"
              << "Wsize: " << _wsize << "\n";
}

void denseStereo::DisparityImage(const cv::Mat &recl, const cv::Mat &recr, cv::Mat &disp, cv::Mat &depth_map) {
    cv::Mat disp16s;
    int N = _ndisp, W = _wsize, C = recl.channels();
    if (is_sgbm) {
        cv::Ptr<cv::StereoSGBM> sgbm =
            cv::StereoSGBM::create(0, N, W, 8 * C * W * W, 32 * C * W * W, 0, 0, 0, 0, 0, cv::StereoSGBM::MODE_SGBM);
        sgbm->compute(recl, recr, disp16s);
    } else {
        cv::Mat grayl, grayr;
        cv::cvtColor(recl, grayl, cv::COLOR_BGR2GRAY);
        cv::cvtColor(recr, grayr, cv::COLOR_BGR2GRAY);

        cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(N, W);
        sbm->setPreFilterCap(31);
        sbm->setMinDisparity(0);
        sbm->setTextureThreshold(10);
        sbm->setUniquenessRatio(15);
        sbm->setSpeckleWindowSize(100);
        sbm->setSpeckleRange(32);
        sbm->setDisp12MaxDiff(1);
        sbm->compute(grayl, grayr, disp16s);
    }

    double minVal, maxVal;
    minMaxLoc(disp16s, &minVal, &maxVal);
    disp16s.convertTo(disp, CV_8UC1, 255 / (maxVal - minVal));

    // How to get the depth map
    double fx = Knew.at<double>(0, 0);
    double bl = -Translation.at<double>(0, 0);

    cv::Mat dispf;
    disp16s.convertTo(dispf, CV_32F, 1.f / 16.f);
    depth_map = cv::Mat(dispf.rows, dispf.cols, CV_32F);

    for (int r = 0; r < dispf.rows; ++r) {
        for (int c = 0; c < dispf.cols; ++c) {

            double disp = dispf.at<float>(r, c);
            if (disp <= 0.f) {
                depth_map.at<float>(r, c) = 0.f;
            } else {
                double depth              = fx * bl / disp;
                depth_map.at<float>(r, c) = static_cast<float>(depth);
            }
        }
    }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr denseStereo::pcFromDepthMap(const cv::Mat &depth_map) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZ>);
    double f_x = Knew.at<double>(0, 0);
    double f_y = Knew.at<double>(1, 1);
    double c_x = Knew.at<double>(0, 2);
    double c_y = Knew.at<double>(1, 2);

    for (int r = 0; r < depth_map.rows; ++r) {
        for (int c = 0; c < depth_map.cols; ++c) {

            double z = static_cast<double>(depth_map.at<float>(r, c));

            // For "foire à la saucisse"
            if (z > 10 || z < 2)
                continue;

            double x = (double)(c - c_x) / f_x;
            double y = (double)(r - c_y) / f_y;

            cv::Vec3d ptcv(x, y, 1);
            double nptcv = cv::norm(ptcv, cv::NORM_L2);
            pcl::PointXYZ pt;
            pt.x = x * z / nptcv;
            pt.y = y * z / nptcv;
            pt.z = z / nptcv;

            // For "foire à la saucisse"
            // ptcv = ptcv * z / nptcv;
            // cv::Vec3d n(0, std::cos(22 * M_PI / 180), std::sin(22 * M_PI / 180));
            // double proj = std::abs(ptcv.dot(n));
            // if (proj > 0.8)
            //     continue;

            pointcloud->points.push_back(pt);
        }
    }

    return pointcloud;
}
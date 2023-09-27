#include <iostream>
#include <fstream>
#include <string>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

#include "vins_logic.h"

inline std::string id2fileName(int id) {
    std::string img_name = "0000000000";
    std::string img_id_str = std::to_string(id);
    return img_name.substr(0, 10 - img_id_str.size()).append(img_id_str);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        LOG(ERROR) << "argc:" << argc << ", argc should be 4";
        return -1;
    }

    vins::init(nullptr, nullptr);

    std::string img_folder_path = argv[0];
    std::string img_time_stamp_path = argv[1];
    std::string imu_folder_path = argv[2];
    std::string imu_time_stamp_path = argv[3];

    if (img_folder_path.back() != '/') {
        img_folder_path.append("/");
    }
    if (img_folder_path.back() != '/') {
        img_folder_path.append("/");
    }

    std::ifstream img_time_stamp_stream(img_time_stamp_path, std::ios::binary);
    std::ifstream imu_time_stamp_stream(imu_time_stamp_path, std::ios::binary);

    std::queue<double> imu_time_stamps;
    std::queue<Eigen::Vector3d> acc_vec, gyr_vec;
    std::string imu_time_stamp_line;
    for (int i = 0; true; ++i) {
        if (!getline(imu_time_stamp_stream, imu_time_stamp_line)) {
            break;
        }
        std::string imu_time_stamp_str = imu_time_stamp_line.substr(17);
        imu_time_stamps.push(std::stod(imu_time_stamp_str));
        std::string imu_file_name = id2fileName(i);
        std::ifstream imu_stream(imu_folder_path + imu_file_name + ".txt", std::ios::binary);
        double lat, lon, alt, roll, pitch, yaw;
        double vn, ve, vf, vl, vu, ax, ay, az, af, al, au, wx, wy, wz, wf, wl, wu;
        double pos_accuracy, vel_accuracy, navstat, numsats, posmode, velmode, orimode;
        imu_stream >> lat >> lon >> alt >> roll >> pitch >> yaw >> vn >>  ve >>  vf >>  vl >>  vu
                    >> ax >> ay >> az >> af >> al >> au >> wx >> wy >> wz >> wf >> wl >> wu
                    >> pos_accuracy >> vel_accuracy >> navstat >> numsats >> posmode >> velmode >> orimode;
        Eigen::Vector3d acc;
        acc << ax, ay, az;
        Eigen::Vector3d gyr;
        gyr << wx, wy, wz;
        acc_vec.push(acc);
        gyr_vec.push(gyr);
    }

    std::string img_time_stamp_line;
    for (int i = 0; true; ++i) {
        if (!getline(img_time_stamp_stream, img_time_stamp_line)) {
            break;
        }
        std::string img_time_stamp_str = img_time_stamp_line.substr(17);
        double img_time_stamp = std::stod(img_time_stamp_str);
        while (!imu_time_stamps.empty() && imu_time_stamps.front() < img_time_stamp) {
            vins::handleIMU(acc_vec.front(), gyr_vec.front(), imu_time_stamps.front());
            acc_vec.pop();
            gyr_vec.pop();
            imu_time_stamps.pop();
        }
        std::shared_ptr<cv::Mat> img = std::make_shared<cv::Mat>();
        *img = cv::imread(img_folder_path + id2fileName(i) + ".png");
        vins::handleImage(img, img_time_stamp);
    }
    return 0;
}
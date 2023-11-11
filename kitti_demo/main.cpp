#include <iostream>
#include <fstream>
#include <string>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

#include "vins/vins_logic.h"

inline std::string id2fileName(int id) {
    std::string img_name = "0000000000";
    std::string img_id_str = std::to_string(id);
    return img_name.substr(0, 10 - img_id_str.size()).append(img_id_str);
}

class Callback : public vins::Callback{
    void onPosSolved(const std::vector<vins::PosAndTimeStamp> & pos_and_time_stamps) {
        LOG(INFO) << pos_and_time_stamps.size() << "\n";
    }
};

std::string join_path(const std::string& path) {
    return path;
}

template <typename ... Args>
std::string join_path(const std::string &t, const Args&... args) {
    assert(!t.empty());
    std::string sum = t;
    if (t.back() != '/') {
        sum += '/';
    }
    sum += join_path(args...);
    return sum;
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    auto callback = std::make_shared<Callback>();
    vins::Param param;
    param.camera.row = 512;
    param.camera.col = 1392;
    vins::init(param, callback);

    std::string data_set_path = "/Users/gjt/vins-mono/dataset";
    std::string img_folder_path = join_path(data_set_path, "image/data");
    std::string img_time_stamp_path = join_path(data_set_path, "image/timestamps.txt");
    std::string imu_folder_path = join_path(data_set_path, "oxts/data");
    std::string imu_time_stamp_path = join_path(data_set_path, "oxts/timestamps.txt");

    std::ifstream img_time_stamp_stream(img_time_stamp_path, std::ios::binary);
    std::ifstream imu_time_stamp_stream(imu_time_stamp_path, std::ios::binary);

    std::queue<double> imu_time_stamps;
    std::queue<Eigen::Vector3d> acc_vec, gyr_vec;
    std::string imu_time_stamp_line;
    LOG(INFO) << "std::string imu_time_stamp_line";
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
        // cv::goodFeaturesToTrack必须是灰度图
        *img = cv::imread(join_path(img_folder_path, id2fileName(i) + ".png"), cv::IMREAD_GRAYSCALE);
        vins::handleImage(img, img_time_stamp);
    }
    int a;
    scanf("%d", &a);
    return 0;
}

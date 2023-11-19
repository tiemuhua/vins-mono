//
// Created by gjt on 11/19/23.
//

#include "visual_inertial_aligner.h"

void test () {
    std::vector<Eigen::Matrix3d> my_img_rots;
    std::vector<Eigen::Matrix3d> my_imu_rots;
    Eigen::Quaterniond ric(1,2,3,4);
    ric.normalize();
    for (int i = 0; i < 10; ++i) {
        Eigen::Quaterniond rot(i*10+1, i*10+2, i*10+3, i*10+4);
        my_img_rots.emplace_back(rot.toRotationMatrix());
        my_imu_rots.emplace_back(ric.toRotationMatrix() * rot.toRotationMatrix() * ric.toRotationMatrix().transpose());
    }
    Eigen::Matrix3d ric1;
    vins::estimateRIC(my_img_rots, my_imu_rots, ric1);
    assert((ric.toRotationMatrix() - ric1).norm() < 1e-5);
}
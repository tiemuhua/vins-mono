//
// Created by gjt on 6/10/23.
//

#ifndef GJT_VINS_PARAMETERS_H
#define GJT_VINS_PARAMETERS_H

#include <string>

namespace vins {
    struct CameraParam {
        int col = -1;
        int row = -1;
        double focal = -1;
    };
    struct FrameTrackerParam {
        double fundamental_threshold{};
        int min_dist{};
        int max_cnt = 100;
    };
    struct FrontEndOptimizeParam {
        bool fix_extrinsic = false;
        bool estimate_time_delay = true;
        int max_iter_num = 100;
    };
    struct IMUParam {
        double ACC_N = 2.0000e-3;
        double ACC_W = 5.0000e-3;
        double GYR_N = 1.6968e-04;
        double GYR_W = 0.003491;
    };

    class Param {
    public:
        CameraParam camera;
        FrameTrackerParam frame_tracker{};
        FrontEndOptimizeParam slide_window;
        IMUParam imu_param{};

        double time_rolling_shatter = 0;
        int window_size = 10;
        double key_frame_parallax_threshold = 0;

        [[nodiscard]] double getTimeShatPerRol() const {
            return time_rolling_shatter / camera.row;
        }
    };

}

#endif //GJT_VINS_PARAMETERS_H
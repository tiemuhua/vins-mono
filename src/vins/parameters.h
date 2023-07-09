//
// Created by gjt on 6/10/23.
//

#ifndef GJT_VINS_PARAMETERS_H
#define GJT_VINS_PARAMETERS_H

#include <string>

namespace vins {
    struct CameraParam{
        int col;
        int row;
        int focal;
    };
    struct FrameTrackerParam{
        double fundamental_threshold;
        int min_dist;
        int max_cnt;
    };
    struct SlideWindowEstimatorParam{
        bool fix_extrinsic = false;
        bool estimate_time_delay = true;
        int max_iter_num = 100;
    };

    class Param{
    public:
        CameraParam camera;
        FrameTrackerParam frame_tracker;

        double gravity_norm;
        double time_rolling_shatter;
        int window_size;
        std::string pattern_file;

        double getTimeShatPerRol() {
            return time_rolling_shatter / camera.row;
        }

        static Param Instance() {
            return param_;
        }

    private:
        static Param param_;
    };

}

#endif //GJT_VINS_PARAMETERS_H

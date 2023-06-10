//
// Created by gjt on 6/10/23.
//

#ifndef GJT_VINS_VINS_DATA_H
#define GJT_VINS_VINS_DATA_H

#include <string>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Eigen>
#include "loop_closer/ThirdParty/DVision/BRIEF.h"
#include <vins/camera_model/include/camodocal/camera_models/Camera.h>

namespace vins {
    class BriefExtractor {
    public:
        BriefExtractor(const std::string &pattern_file) {
            // The DVision::BRIEF extractor computes a random pattern by default when
            // the object is created.
            // We load the pattern that we used to build the vocabulary, to make
            // the descriptors compatible with the predefined vocabulary

            // loads the pattern
            cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
            if (!fs.isOpened()) throw std::string("Could not open file ") + pattern_file;

            std::vector<int> x1, y1, x2, y2;
            fs["x1"] >> x1;
            fs["x2"] >> x2;
            fs["y1"] >> y1;
            fs["y2"] >> y2;

            m_brief.importPairs(x1, y1, x2, y2);
        }
        DVision::BRIEF m_brief;
    };

    class RunInfo {
    public:
        BriefExtractor extractor;
        Eigen::Vector3d tic;
        Eigen::Matrix3d ric;
        camodocal::CameraPtr camera_ptr;
        static RunInfo& Instance() {
            return run_info_;
        }
    private:
        static RunInfo run_info_;
    };

}

#endif //GJT_VINS_VINS_DATA_H

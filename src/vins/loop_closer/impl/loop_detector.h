//
// Created by gjt on 7/9/23.
//

#ifndef GJT_VINS_LOOP_DETECTOR_H
#define GJT_VINS_LOOP_DETECTOR_H

#include "DBoW2/DBoW2.h"

namespace vins{
    class LoopDetector {
    public:
        LoopDetector(const std::string &voc_path) {
            voc = new BriefVocabulary(voc_path);
            db.setVocabulary(*voc, false, 0);
        }
        /**
         * 成功建立回环则返回之前帧的id，否则返回-1
         * */
        [[nodiscard]] int detectLoop(const std::vector<DVision::BRIEF::bitset>& descriptors, int frame_index) const;
        void addDescriptors(const std::vector<DVision::BRIEF::bitset>& descriptors)  {
            db.add(descriptors);
        }
    private:
        BriefDatabase db;
        BriefVocabulary *voc{};
    };

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

            brief_.importPairs(x1, y1, x2, y2);
        }
        DVision::BRIEF brief_;
    };
}


#endif //GJT_VINS_LOOP_DETECTOR_H

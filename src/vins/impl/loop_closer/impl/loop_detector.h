//
// Created by gjt on 7/9/23.
//

#ifndef GJT_VINS_LOOP_DETECTOR_H
#define GJT_VINS_LOOP_DETECTOR_H

#include "DBoW2/DBoW2.h"

namespace vins {
    class LoopDetector {
    public:
        LoopDetector(const std::string &voc_path) {
            voc = new BriefVocabulary(voc_path);
            db.setVocabulary(*voc, false, 0);
        }

        /**
         * 成功建立回环则返回之前帧的id，否则返回-1
         * */
        [[nodiscard]] int
        detectSimilarDescriptor(const std::vector<DVision::BRIEF::bitset> &descriptors, int frame_index) const;

        void addDescriptors(const std::vector<DVision::BRIEF::bitset> &descriptors) {
            db.add(descriptors);
        }

    private:
        BriefDatabase db;
        BriefVocabulary *voc{};
    };
}


#endif //GJT_VINS_LOOP_DETECTOR_H

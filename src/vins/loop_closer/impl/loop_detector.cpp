//
// Created by gjt on 7/9/23.
//

#include "loop_detector.h"

#include "DVision/DVision.h"
#include "DBoW2/TemplatedVocabulary.h"
#include <DBoW2/QueryResults.h>

using namespace DBoW2;
using namespace vins;

int LoopDetector::detectSimilarDescriptor(const std::vector<DVision::BRIEF::bitset>& descriptors, int frame_index) const {
    if (frame_index < 50) {
        return -1;
    }
    QueryResults ret;
    db.query(descriptors, ret, 4, frame_index - 50);
    cv::Mat loop_result;
    // a good match with its neighbour
    if (ret.size() < 2 || ret[0].Score < 0.05) {
        return -1;
    }
    bool find_loop = false;
    for (unsigned int i = 1; i < ret.size(); i++) {
        if (ret[i].Score > 0.015) {
            find_loop = true;
        }
    }
    if (!find_loop) {
        return -1;
    }
    int min_index = 0x3f3f3f3f;
    assert(ret.size() < min_index);
    for (unsigned int i = 1; i < ret.size(); i++) {
        if (ret[i].Id < min_index && ret[i].Score > 0.015)
            min_index = ret[i].Id;
    }
    return min_index;
}

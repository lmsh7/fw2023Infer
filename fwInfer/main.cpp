//
// Created by LMSH7 on 2023/4/17.
//
#include <glog/logging.h>
#include "data/tensor.hpp"

int main() {
    LOG(INFO) << "Kuiper Infer Course";
    auto t = new fw_infer::Tensor<float>(1, 2, 3);
    return 0;
}

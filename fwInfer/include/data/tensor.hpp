//
// Created by LMSH7 on 2023/4/17.
//

#ifndef FWINFER_TENSOR_HPP
#define FWINFER_TENSOR_HPP

#include <armadillo>

namespace fw_infer {

    template<typename T>
    class Tensor {

    };

    template<>
    class Tensor<float> {
    public:
        explicit Tensor() = default;

        explicit Tensor(uint32_t channel, uint32_t height, uint32_t width);
    };
}

#endif //FWINFER_TENSOR_HPP

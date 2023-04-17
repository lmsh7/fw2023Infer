#include "data/tensor.hpp"
#include <gtest/gtest.h>
#include <glog/logging.h>

#include "data/tensor.hpp"
#include <gtest/gtest.h>
#include <armadillo>
#include <glog/logging.h>
#include "data/tensor.hpp"
#include <iostream>

TEST(test_tensor, one) {
    using namespace fw_infer;
    Tensor<float> tensor(3, 4, 5);
    tensor.Ones();
    for (int i = 0; i < tensor.channels(); ++i) {
        for (int j = 0; j < tensor.rows(); ++j) {
            for (int k = 0; k < tensor.cols(); ++k) {
                ASSERT_EQ(tensor.at(i, j, k), 1);
            }
        }
    }
}

TEST(test_tensor, create) {
    using namespace fw_infer;
    Tensor<float> tensor(3, 32, 32);
    ASSERT_EQ(tensor.channels(), 3);
    ASSERT_EQ(tensor.rows(), 32);
    ASSERT_EQ(tensor.cols(), 32);
    ASSERT_EQ(tensor.empty(), false);
}

TEST(test_tensor, fill) {
    using namespace fw_infer;
    Tensor<float> tensor(3, 3, 3);
    ASSERT_EQ(tensor.channels(), 3);
    ASSERT_EQ(tensor.rows(), 3);
    ASSERT_EQ(tensor.cols(), 3);

    std::vector<float> values;
    for (int i = 0; i < 27; ++i) {
        values.push_back((float) i);
    }
    tensor.Fill(values);
    LOG(INFO) << tensor.data();

    int index = 0;
    for (int c = 0; c < tensor.channels(); ++c) {
        for (int c_ = 0; c_ < tensor.cols(); ++c_) {
            for (int r = 0; r < tensor.rows(); ++r) {
                ASSERT_EQ(values.at(index), tensor.at(c, c_, r));
                index += 1;
            }
        }
    }
    LOG(INFO) << "Test1 passed!";
}

TEST(test_tensor, padding1) {
    using namespace fw_infer;
    Tensor<float> tensor(3, 3, 3);
    ASSERT_EQ(tensor.channels(), 3);
    ASSERT_EQ(tensor.rows(), 3);
    ASSERT_EQ(tensor.cols(), 3);

    tensor.Fill(1.f); // 填充为1
    tensor.Padding({1, 1, 1, 1}, 0); // 边缘填充为0
    ASSERT_EQ(tensor.rows(), 5);
    ASSERT_EQ(tensor.cols(), 5);

    int index = 0;
    // 检查一下边缘被填充的行、列是否都是0
    for (int c = 0; c < tensor.channels(); ++c) {
        for (int c_ = 0; c_ < tensor.cols(); ++c_) {
            for (int r = 0; r < tensor.rows(); ++r) {
                if (c_ == 0 || r == 0) {
                    ASSERT_EQ(tensor.at(c, c_, r), 0);
                }
                index += 1;
            }
        }
    }
    LOG(INFO) << "Test2 passed!";
}

TEST(test_tensor, flatten) {
    using namespace fw_infer;
    uint32_t channel = 1;
    uint32_t rows = 3;
    uint32_t cols = 3;

    Tensor<float> tensor(channel, rows, cols);
    tensor.Rand();
    Tensor<float> origin_tensor = tensor;
    tensor.Flatten();

    tensor.Show();

    uint32_t flatten_size = tensor.channels() * tensor.rows() * tensor.cols();

    for (uint32_t i = 0; i < flatten_size; i++) {
        // 验证展平后的数据是否与原始数据一致
        float value = tensor.index(i);
        float raw_value = origin_tensor.at(i / (rows * cols), (i % (rows * cols)) / cols, i % cols);
        ASSERT_EQ(value, raw_value);
    }

    std::vector<uint32_t> expected_shapes = {1, flatten_size, 1};
    ASSERT_EQ(expected_shapes, tensor.shapes());

}

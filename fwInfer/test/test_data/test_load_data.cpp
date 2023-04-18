//
// Created by LMSH7 on 2023/4/17.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include <armadillo>
#include "data/load_data.hpp"
#include <filesystem>

TEST(test_load, load_csv_data) {
    using namespace fw_infer;
    std::shared_ptr<Tensor<float>> data = CSVDataLoader::LoadData("../../test/tmp/data_loader/data1.csv");
    ASSERT_NE(data->empty(), true);
    ASSERT_EQ(data->rows(), 3);
    ASSERT_EQ(data->cols(), 4);
    const uint32_t rows = data->rows();
    const uint32_t cols = data->cols();
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            ASSERT_EQ(data->at(0, i, j), 1);
        }
    }
}

TEST(test_load, load_csv_arange) {
    using namespace fw_infer;
    std::shared_ptr<Tensor<float>> data = CSVDataLoader::LoadData("../../test/tmp/data_loader/data2.csv");
    ASSERT_NE(data->empty(), true);
    ASSERT_EQ(data->rows(), 3);
    ASSERT_EQ(data->cols(), 4);

    int range_data = 0;
    const uint32_t rows = data->rows();
    const uint32_t cols = data->cols();
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            ASSERT_EQ(data->at(0, i, j), range_data);
            range_data += 1;
        }
    }
}

TEST(test_load, load_csv_missing_data1) {
    using namespace fw_infer;
    std::shared_ptr<Tensor<float>> data = CSVDataLoader::LoadData("../../test/tmp/data_loader/data4.csv");
    ASSERT_NE(data->empty(), true);
    ASSERT_EQ(data->rows(), 3);
    ASSERT_EQ(data->cols(), 11);
    int data_one = 0;
    const uint32_t rows = data->rows();
    const uint32_t cols = data->cols();
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            if (data->at(0, i, j) == 1) {
                data_one += 1;
            }
        }
    }
    ASSERT_EQ(data->at(0, 2, 1), 0);
    ASSERT_EQ(data_one, 32);
}
//
TEST(test_load, load_csv_missing_data2) {
    using namespace fw_infer;
    std::shared_ptr<Tensor<float>> data = CSVDataLoader::LoadData("../../test/tmp/data_loader/data3.csv");

    ASSERT_NE(data->empty(), true);
    ASSERT_EQ(data->rows(), 3);
    ASSERT_EQ(data->cols(), 11);

    const uint32_t rows = data->rows();
    const uint32_t cols = data->cols();
    int data_one = 0;
    int data_zero = 0;
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            if (data->at(0, i, j) == 1) {
                data_one += 1;
            } else if (data->at(0, i, j) == 0) {
                data_zero += 1;
            }
        }
    }
    ASSERT_EQ(data->at(0, 2, 10), 0);
    ASSERT_EQ(data_zero, 1);
    ASSERT_EQ(data_one, 32);
}

TEST(test_load, split_char) {
    using namespace fw_infer;
    std::shared_ptr<Tensor<float>> data = CSVDataLoader::LoadData("../../test/tmp/data_loader/data5.csv", '-');

    ASSERT_NE(data->empty(), true);
    ASSERT_EQ(data->rows(), 3);
    ASSERT_EQ(data->cols(), 11);

    const uint32_t rows = data->rows();
    const uint32_t cols = data->cols();
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            ASSERT_EQ(data->at(0, i, j), 1);
        }
    }
}

TEST(test_load, load_minus_data) {
    using namespace fw_infer;
    std::shared_ptr<Tensor<float>> data = CSVDataLoader::LoadData("../../test/tmp/data_loader/data6.csv", ',');

    ASSERT_NE(data->empty(), true);
    ASSERT_EQ(data->rows(), 3);
    ASSERT_EQ(data->cols(), 11);

    int data_minus_one = 0;

    const uint32_t rows = data->rows();
    const uint32_t cols = data->cols();
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            if (data->at(0, i, j) == -1) {
                data_minus_one += 1;
            }
        }
    }
    ASSERT_EQ(data_minus_one, 33);
}

TEST(test_load, load_large_data) {
    using namespace fw_infer;
    std::shared_ptr<Tensor<float>> data = CSVDataLoader::LoadData("../../test/tmp/data_loader/data7.csv", ',');
    ASSERT_NE(data->empty(), true);
    ASSERT_EQ(data->rows(), 1024);
    ASSERT_EQ(data->cols(), 1024);

    const uint32_t rows = data->rows();
    const uint32_t cols = data->cols();

    int data_minus_one = 0;
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            if (data->at(0, i, j) == -1) {
                data_minus_one += 1;
            }
        }
    }
    ASSERT_EQ(data_minus_one, 1024 * 1024);
}

TEST(test_load, load_empty_data) {
    using namespace fw_infer;
    // catch error when file not exists
    ASSERT_DEATH(CSVDataLoader::LoadData("../../test/tmp/data_loader/nonexistent.csv", ','), ".*");
}

TEST(test_data_load, load_csv_with_head1) {
    using namespace fw_infer;
    const std::string &file_path = "../../test/tmp/data_loader/data8.csv";
    std::vector<std::string> headers;
    std::shared_ptr<Tensor<float>> data = CSVDataLoader::LoadDataWithHeader(file_path, headers, ',');

    uint32_t index = 1;
    uint32_t rows = data->rows();
    uint32_t cols = data->cols();
    LOG(INFO) << "\n" << data;
    ASSERT_EQ(rows, 3);
    ASSERT_EQ(cols, 3);
    ASSERT_EQ(headers.size(), 3);

    ASSERT_EQ(headers.at(0), "ROW1");
    ASSERT_EQ(headers.at(1), "ROW2");
    ASSERT_EQ(headers.at(2), "ROW3");

    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            ASSERT_EQ(data->at(0, r, c), index);
            index += 1;
        }
    }
}
//
// Created by LMSH7 on 2023/4/17.
//


#ifndef FWINFER_LOAD_DATA_HPP
#define FWINFER_LOAD_DATA_HPP

#include <armadillo>
#include <string>
#include "data/tensor.hpp"

namespace fw_infer {
    class CSVDataLoader {
    public:
        static std::shared_ptr<Tensor<float >> LoadData(
                const std::string &file_path,
                char split_char = ','
        );

        static std::shared_ptr<Tensor<float >> LoadDataWithHeader(
                const std::string &file_path,
                std::vector<std::string>
                &headers,
                char split_char = ','
        );

    private:
        static std::pair<size_t, size_t> GetMatrixSize(std::ifstream &file, char split_char);
    };
}
#endif //FWINFER_LOAD_DATA_HPP


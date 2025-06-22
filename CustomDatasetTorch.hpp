#ifndef CUSTOM_DATASET_TORCH_HPP
#define CUSTOM_DATASET_TORCH_HPP

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class CustomDatasetTorch : public torch::data::Dataset<CustomDatasetTorch> {
private:
    std::vector<std::pair<std::string, int64_t>> samples;
    std::string data_root;
    int image_size;
    bool is_rgb;

    void load_csv(const std::string& csv_path);

public:
    CustomDatasetTorch(const std::string& root_dir, int img_size, bool is_testing = false, bool rgb = false);
    
    torch::data::Example<> get(size_t index) override;
    
    torch::optional<size_t> size() const override;
};

#endif // CUSTOM_DATASET_TORCH_HPP 
#include "CustomDatasetTorch.hpp"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <stdexcept>

void CustomDatasetTorch::load_csv(const std::string& csv_path) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + csv_path);
    }

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string filename;
        int64_t label;
        
        if (std::getline(ss, filename, ',') && (ss >> label)) {
            samples.emplace_back(filename, label);
        }
    }
}

CustomDatasetTorch::CustomDatasetTorch(const std::string& root_dir, int img_size, bool is_testing, bool rgb) 
    : data_root(root_dir), image_size(img_size), is_rgb(rgb) {
    std::string csv_path = data_root + "/" + (is_testing ? "test.csv" : "train.csv");
    load_csv(csv_path);
}

torch::data::Example<> CustomDatasetTorch::get(size_t index) {
    const auto& [rel_path, label] = samples[index];
    std::string full_path = data_root + "/" + rel_path;

    cv::Mat img;
    if (is_rgb) {
        img = cv::imread(full_path, cv::IMREAD_COLOR);
        if (!img.empty()) {
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        }
    } else {
        img = cv::imread(full_path, cv::IMREAD_GRAYSCALE);
    }

    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + full_path);
    }

    cv::resize(img, img, cv::Size(image_size, image_size));
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    torch::Tensor tensor_image;
    if (is_rgb) {
        tensor_image = torch::from_blob(img.data, {image_size, image_size, 3}, torch::kFloat32);
        tensor_image = tensor_image.permute({2, 0, 1});
    } else {
        tensor_image = torch::from_blob(img.data, {image_size, image_size}, torch::kFloat32);
        tensor_image = tensor_image.unsqueeze(0);
    }

    tensor_image = tensor_image.clone();
    return {tensor_image, torch::tensor(label)};
}

torch::optional<size_t> CustomDatasetTorch::size() const {
    return samples.size();
} 
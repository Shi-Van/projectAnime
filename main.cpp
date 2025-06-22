#include "CustomDatasetTorch.hpp"
#include <iostream>

int main() {
    try {
        // Создаём датасет MNIST (тренировочный)
        CustomDatasetTorch dataset("data/mnist", 28, false, false);
        
        // Проверяем размер датасета
        auto size = dataset.size();
        if (!size.has_value()) {
            std::cerr << "Ошибка: размер датасета не определён\n";
            return 1;
        }
        std::cout << "Размер датасета: " << size.value() << " изображений\n";

        // Проверяем первые 5 элементов
        for (size_t i = 0; i < 5; ++i) {
            auto example = dataset.get(i);
            auto label = example.target.item<int64_t>();
            
            std::cout << "Изображение " << i << ":\n";
            std::cout << "  Метка: " << label << "\n";
            std::cout << "  Размер тензора: " << example.data.sizes() << "\n";
        }

        std::cout << "Тест успешно завершён!\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }
}

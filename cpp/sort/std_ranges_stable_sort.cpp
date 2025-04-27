#include <array>
#include <format>
#include <iostream>
#include <algorithm>

int main() noexcept {
    std::array<int,12> arr = {12,24,534,2,123,5,3,7,2,-123,0,23};
    std::cout << "before:\t{";
    for(const auto& item : arr)
        std::cout << std::format("{},",item);
    std::cout << "\b}" << std::endl;

    std::ranges::stable_sort(arr);

    std::cout << "after:\t{";
    for(const auto& item : arr)
        std::cout << std::format("{},",item);
    std::cout << "\b}" << std::endl;
    
    return 0;
}
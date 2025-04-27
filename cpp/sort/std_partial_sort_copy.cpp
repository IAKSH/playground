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

    std::array<int,12> res;
    // std::partial_sort_copy is mostly the same of std::partial_sort
    // partial_sort_copy copies sorted elems to some other (or the same?) location
    std::partial_sort_copy(std::begin(arr),std::end(arr),std::begin(res),std::end(res));
    // skip

    std::cout << "after:\t{";
    for(const auto& item : arr)
        std::cout << std::format("{},",item);
    std::cout << "\b}" << std::endl;

    std::cout << "res:\t{";
    for(const auto& item : res)
        std::cout << std::format("{},",item);
    std::cout << "\b}" << std::endl;
    
    return 0;
}
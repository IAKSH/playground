#include <array>
#include <format>
#include <ranges>
#include <iostream>
#include <algorithm>

int main() noexcept {
    std::array<int,12> arr = {12,24,534,2,123,5,3,7,2,-123,0,23};
    std::cout << "before:\t{";
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        std::cout << std::format("{},",arr[i]);
    std::cout << "\b}" << std::endl;

    //std::make_heap(std::begin(arr),std::end(arr));
    //std::sort_heap(std::begin(arr),std::end(arr));
    std::ranges::make_heap(arr);
    std::ranges::sort_heap(arr);

    std::cout << "after:\t{";
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        std::cout << std::format("{},",arr[i]);
    std::cout << "\b}" << std::endl;
    
    return 0;
}
#include <array>
#include <format>
#include <iostream>
#include <algorithm>

int main() noexcept {
    std::array<int,12> arr = {12,24,534,2,123,5,3,7,2,-123,0,23};
    std::cout << "before:\t\t{";
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        std::cout << std::format("{},",arr[i]);
    std::cout << "\b}" << std::endl;

    std::array<int,12> ret_arr;
    // std::partial_sort_copy is mostly the same of std::partial_sort
    // partial_sort_copy copies sorted elems to some other (or the same?) location
    std::partial_sort_copy(std::begin(arr),std::end(arr),std::begin(ret_arr),std::end(ret_arr));
    // skip

    std::cout << "after:\t\t{";
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        std::cout << std::format("{},",arr[i]);
    std::cout << "\b}" << std::endl;

    std::cout << "ret_arr:\t{";
    for(int i = 0;i < sizeof(ret_arr)/sizeof(int);i++)
        std::cout << std::format("{},",ret_arr[i]);
    std::cout << "\b}" << std::endl;
    
    return 0;
}
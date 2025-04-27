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

    // std::partial_sort is actually using heap sort
    // it will sort from first to middle, thus "partial"
    // partial sort maybe sometimes faster than std::sort (really?)
    // okay, I this it's value of gramma is far more than performance
    std::partial_sort(std::begin(arr),std::begin(arr) + arr.size() / 2,std::end(arr));

    // you can modify how std::partial_sort compare each elem
    //std::partial_sort(std::begin(arr),std::begin(arr) + arr.size() / 2,std::end(arr),std::less<int>());
    /*
    std::partial_sort(std::begin(arr),std::begin(arr) + arr.size() / 2,std::end(arr),[](const int& a,const int& b) {
        return a < b;
    });
    // if you need RVO, you can also use copy in the lambda
    */

    std::cout << "after:\t{";
    for(const auto& item : arr)
        std::cout << std::format("{},",item);
    std::cout << "\b}" << std::endl;
    
    return 0;
}
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

    // std::sort is actually using "intro sort" (which comes from quick sort but can handle all cases with O(N·log(N)) comparisons)
    // qucik sort may need O(N^2) comparisons in the worst case
    // intro sort normally use quick sort at first, and transform to heap sort if it goes too deep in recursion
    std::sort(std::begin(arr),std::end(arr));
    
    // you can modify how std::sort compare each elem
    //std::sort(std::begin(arr),std::end(arr),std::less<int>());
    /*
    std::sort(std::begin(arr),std::end(arr),[](const int& a,const int& b) {
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
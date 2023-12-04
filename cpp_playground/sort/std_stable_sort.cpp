#include <array>
#include <format>
#include <iostream>
#include <algorithm>

int main() noexcept {
    std::array<int,12> arr = {12,24,534,2,123,5,3,7,2,-123,0,23};
    std::cout << "before:\t{";
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        std::cout << std::format("{},",arr[i]);
    std::cout << "\b}" << std::endl;

    // std::stable_sort is acutally using merge sort
    // which is stable but may need more memory
    // if not, it's complexity will increase from O(N·log(N)) to O(N·log^2(N))
    // however, we can't control whether std::stable_sort use extral memory or not
    // this was decided inside it's implementation
    std::stable_sort(std::begin(arr),std::end(arr));

    // you can modify how std::stable_sort compare each elem
    //std::stable_sort(std::begin(arr),std::end(arr),std::less<int>());
    /*
    std::stable_sort(std::begin(arr),std::end(arr),[](int a,int b) {
        return a < b;
    });
    // if you need RVO, you can also use copy in the lambda
    */

    std::cout << "after:\t{";
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        std::cout << std::format("{},",arr[i]);
    std::cout << "\b}" << std::endl;
    
    return 0;
}
#include <iostream>
#include <array>
#include <ranges>

int main()
{
    std::array<int, 10> arr{514,114,1215,211,12,2,48,695,6594,125};
    std::ranges::sort(arr);
    for(const auto& item : arr)
        std::cout << item << '\t';

    std::cout << std::endl;
    return 0;
}
#include <iostream>
#include <array>
#include <ranges>

int main()
{
    std::array<int, 10> arr{514,114,1215,211,12,2,48,695,6594,125};

    auto even = [](const int& a)
    {
        return a % 2 == 0;
    };

    auto v = arr | std::views::filter(even)
        | std::views::transform([](const int& a) {return a * a; })
        | std::views::take(2);

    std::cout << *v.begin()  << std::endl;

    for(const auto& item : arr)
        std::cout << item << '\t';
    std::cout << std::endl;
    return 0;
}
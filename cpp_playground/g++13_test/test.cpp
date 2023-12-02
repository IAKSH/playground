#include <iostream>
#include <vector>
#include <algorithm>
#include <ranges>

int main() noexcept {
    std::vector<std::vector<int>> vec{
        std::vector<int>{1,1,2,3},
        std::vector<int>{0,21,1,2},
        std::vector<int>{1,1,2,3}
    };
    std::ranges::sort(vec);
    vec.erase(std::unique(std::begin(vec),std::end(vec)),std::end(vec));
    for(const auto& v : vec) {
        std::cout << "{ ";
        for(const auto& i : v)
            std::cout << i << ' ';
        std::cout << "}" << std::endl;
    }
}
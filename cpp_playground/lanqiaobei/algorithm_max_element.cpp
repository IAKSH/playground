#include <iostream>
#include <vector>
#include <algorithm>

// std::min_element的用法和std::max_element一样

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    auto max_it = std::max_element(v.begin(), v.end());

    std::cout << "Max element is " << *max_it
        << " at position " << std::distance(v.begin(), max_it) << std::endl;

    return 0;
}

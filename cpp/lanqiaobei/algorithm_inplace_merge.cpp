#include <iostream>
#include <vector>
#include <algorithm>

/**
 * std::inplace_merge
 * 就是inplace merge，详见algorithm_merge.cpp
*/

int main() {
    std::vector<int> v = {1, 4, 5, 2, 3, 6};

    // 将已排序的两个连续范围合并
    std::inplace_merge(v.begin(), v.begin() + 3, v.end());

    std::for_each(v.begin(), v.end(), [](int x){
        std::cout << x << ' ';
    });
    std::cout << '\n';

    return 0;
}

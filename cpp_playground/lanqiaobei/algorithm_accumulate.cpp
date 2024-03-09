#include <iostream>
#include <vector>
#include <numeric>

/**
 * std::accumulate
 * 用于计算给定范围 [first, last) 中的元素和给定值 init 的总和。
 * 第一个版本使用 operator+ 来累加元素，第二个版本使用给定的二元函数 op。
 * 两者都在左操作数上应用 std::move（自 C++20 起）
*/

int main() {
    std::vector<int> v {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int sum = std::accumulate(v.begin(), v.end(), 0);

    std::cout << "sum: " << sum << '\n';

    return 0;
}

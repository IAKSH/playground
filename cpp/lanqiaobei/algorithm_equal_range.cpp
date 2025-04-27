#include <iostream>
#include <vector>
#include <algorithm>

/*
 * std::equal_range
 * 返回一个范围 [first, last) 中所有等于给定值的元素的子范围的边界
 */

int main() {
    std::vector<int> v = {3, 1, 4, 2, 5};
    std::sort(v.begin(), v.end());

    // 二分查找等于3的元素的子范围
    auto result = std::equal_range(v.begin(), v.end(), 3);

    std::cout << *result.first << std::endl;  // 输出子范围的开始
    std::cout << *result.second << std::endl; // 输出子范围的结束

    return 0;
}

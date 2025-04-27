#include <iostream>
#include <algorithm>
#include <vector>

/*
 * 在已排序的范围中查找特定元素。
 * 返回该元素是否存在 (bool)
 */

int main() {
    std::vector<int> v = {1, 2, 4, 4, 5, 6};

    // 查找元素4是否存在
    bool found = std::binary_search(v.begin(), v.end(), 4);

    if (found) {
        std::cout << "Element found in the vector" << std::endl;
    } else {
        std::cout << "Element not found in the vector" << std::endl;
    }

    return 0;
}

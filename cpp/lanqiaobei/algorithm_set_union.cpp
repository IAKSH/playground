#include <iostream>
#include <algorithm>
#include <vector>

/*
 * std::set_union
 * 用于构造两个已排序范围的并集。
 */

int main() {
    std::vector<int> v1 = {1, 2, 4, 5, 6};
    std::vector<int> v2 = {2, 3, 5, 7, 8};
    std::vector<int> v_union;

    std::set_union(v1.begin(), v1.end(), v2.begin(), v2.end(), std::back_inserter(v_union));

    std::cout << "The union has " << v_union.size() << " elements:\n";
    for (int n : v_union)
        std::cout << ' ' << n;
    std::cout << '\n';

    return 0;
}

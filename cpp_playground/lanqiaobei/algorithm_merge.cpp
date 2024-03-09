#include <iostream>
#include <algorithm>
#include <vector>

/**
 * std::merge
 * 用于合并两个已排序的范围
*/

int main() {
    std::vector<int> v1 = {1, 2, 4, 5, 6};
    std::vector<int> v2 = {2, 3, 5, 7, 8};
    std::vector<int> v_merge;

    std::merge(v1.begin(), v1.end(), v2.begin(), v2.end(), std::back_inserter(v_merge));

    std::cout << "The merged vector has " << v_merge.size() << " elements:\n";
    for (int n : v_merge)
        std::cout << ' ' << n;
    std::cout << '\n';

    return 0;
}

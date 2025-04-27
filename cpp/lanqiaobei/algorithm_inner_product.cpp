#include <numeric>
#include <iostream>
#include <vector>

/**
 * 计算内积（点乘）
*/

int main () {
    std::vector<int> a {0, 1, 2, 3, 4};
    std::vector<int> b {5, 4, 2, 3, 1};

    int r1 = std::inner_product(a.begin(), a.end(), b.begin(), 0);

    std::cout << "Inner product of a and b: " << r1 << '\n';

    return 0;
}

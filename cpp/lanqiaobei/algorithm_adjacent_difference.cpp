#include <iostream> // std::cout
#include <numeric> // std::adjacent_difference

/**
 * std::adjacent_difference
 * 计算区域内相邻元素之间的差异
*/

int main () {
    int val [] = { 5, 7, 4, 8, 2 };
    int n = sizeof(val) / sizeof(val [0]);
    int result [n];

    std::cout << "Array contains :";
    for (int i = 0; i < n; i++)
        std::cout << " " << val [i];
    std::cout << "\n";

    std::adjacent_difference(val, val + n, result);

    std::cout << "Result contains :";
    for (int i = 0; i < n; i++)
        std::cout << ' ' << result [i];
    std::cout << '\n';

    return 0;
}

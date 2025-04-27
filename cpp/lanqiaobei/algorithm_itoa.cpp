#include <iostream>
#include <numeric>

/**
 * std::iota
 * 不是std::itoa
 * 用于在给定范围 [first, last) 中生成连续的值。
 * 这个函数会将 value 和它的连续增量填充到范围 [first, last) 中
*/

int main() {
    int numbers[11];
    int st = 10;

    std::iota(numbers, numbers + 11, st);

    std::cout << "Elements are: ";
    for (auto i : numbers)
        std::cout << ' ' << i;
    std::cout << '\n';

    return 0;
}

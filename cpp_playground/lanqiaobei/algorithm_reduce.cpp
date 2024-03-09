#include <iostream> // std::cout
#include <numeric> // std::reduce
#include <vector> // std::vector

/**
 * std::reduce (C++ 17)
 * 实际上是执行顺序不确定的std::accumulate
 * C++ 98的std::accumulate是顺序执行的，而C++ 17的std::reduce可以乱序（进而可以比较方便地实现并行）
 * 虽然具体如何执行，还是看编译器，最坏情况下std::reduce将会和std::accumulate一样顺序执行。
*/

int main () {
#if __cplusplus >= 201703L
    std::vector<int> v = {1, 2, 3, 4, 5};

    int sum = std::reduce(v.begin(), v.end());

    std::cout << "The sum is " << sum << '\n';
#else
    std::cout << "need C++ 17\n";
#endif
    return 0;
}

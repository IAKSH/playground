#include <iostream>  // std::cout
#include <algorithm> // std::stable_partition
#include <vector>    // std::vector

/*
 * 详见algorithm_partition.cpp
 * std::stable_partition中的stable就是排序的stable。毕竟std::partition实际上也是排序。
 */

bool IsOdd(int i) { return (i % 2) == 1; }

int main()
{
    // 在这个例子中，std::stable_partition 将奇数和偶数分开，使得所有的奇数都位于偶数之前，并且保持了元素的相对顺序

    std::vector<int> myvector;
    // set some values:
    for (int i = 1; i < 10; ++i)
        myvector.push_back(i); // 1 2 3 4 5 6 7 8 9

    std::vector<int>::iterator bound;
    bound = std::stable_partition(myvector.begin(), myvector.end(), IsOdd);

    // print out content:
    std::cout << "odd elements:";
    for (std::vector<int>::iterator it = myvector.begin(); it != bound; ++it)
        std::cout << ' ' << *it;
    std::cout << '\n';

    std::cout << "even elements:";
    for (std::vector<int>::iterator it = bound; it != myvector.end(); ++it)
        std::cout << ' ' << *it;
    std::cout << '\n';

    return 0;
}

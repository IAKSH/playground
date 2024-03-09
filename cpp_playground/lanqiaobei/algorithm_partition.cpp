#include <iostream>  // std::cout
#include <algorithm> // std::partition
#include <vector>    // std::vector

/*
 * 将范围 [first, last) 中的元素重新排序，使得所有使谓词 p 返回 true 的元素都位于所有使 p 返回 false 的元素之前。
 * 换句话说，它会将满足特定条件的元素和不满足该条件的元素分开。
 */

int main()
{
    // 在这个例子中，std::partition 将奇数和偶数分开，使得所有的奇数都位于偶数之前。

    std::vector<int> myvector;
    // set some values:
    for (int i = 1; i < 10; ++i)
        myvector.push_back(i); // 1 2 3 4 5 6 7 8 9

    std::vector<int>::iterator bound;
    bound = std::partition(myvector.begin(), myvector.end(), [](int i)
                           { return (i % 2) == 1; });

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

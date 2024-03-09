#include <iostream>
#include <string>
#include <iterator>
#include <random>
#include <algorithm>

/*
 * std::sample
 * since C++ 17
 * 从给定的范围中均匀地选择最多 n 个元素，并将这些选定的元素写入输出迭代器。
 * 随机数可以使用随机数生成器函数生成
 */

int main() {
#if __cplusplus >= 201703L
    // 乱数生成器
    std::random_device seed_gen;
    std::mt19937 engine { seed_gen() };

    // 从字符串中随机抽取3个字符
    {
        const std::string input = "abcdef";
        const int n = 3;
        std::string result;
        std::sample(input.begin(), input.end(), std::back_inserter(result), n, engine);
        std::cout << result << std::endl;
    }

    // 从数组中随机抽取3个元素
    {
        const std::vector<int> input = {0, 1, 2, 3, 4, 5};
        const int n = 3;
        std::vector<int> result;
        std::sample(input.begin(), input.end(), std::back_inserter(result), n, engine);
        for (int x : result) {
            std::cout << x;
        }
        std::cout << std::endl;
    }
#else
    std::cout << "need C++ 17\n";
#endif
}

#include <bits/stdc++.h>

/**
 * std::clamp (C++ 17)
 * 用于将一个值限制在一个给定的范围内。
 * 如果 num 大于 high，num 将被赋值为 high。
 * 如果 num 小于 low，num 将被赋值为 low。
 * 如果 num 已经在范围内，那么 num 不会被修改。
*/

int main () {
#if __cplusplus >= 201703L
    int high = 100, low = 10;
    int num1 = 120;
    int num2 = 5;
    int num3 = 50;

    num1 = std::clamp(num1, low, high);
    num2 = std::clamp(num2, low, high);
    num3 = std::clamp(num3, low, high);

    std::cout << num1 << " " << num2 << " " << num3;
#else
    std::cout << "need C++ 17\n";
#endif

    return 0;
}

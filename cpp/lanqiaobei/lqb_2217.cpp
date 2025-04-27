// https://www.lanqiao.cn/problems/2217/learning/?page=1&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B,2022&sort=pass_rate&asc=0

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    // 只在关于区间内到底有多少次重合这一点上有一些迷惑人
    // 6 [14,60)
    // 7 59
    // 8
    // 9
    // 10
    // 11
    // 12
    // 13
    // 14 [0,36)
    cout << (59 - 13) + 59 * (13 - 6) + 35 << '\n';
    return 0;
}
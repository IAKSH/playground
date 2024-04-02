/**
 * https://www.lanqiao.cn/problems/3493/learning/?page=1&first_category_id=1&second_category_id=3&difficulty=20&tags=2023
 * accumulate 1..20230408
 * 实际上是一个等差数列
 * a_1 = 1
 * a_n = 20230408
 * S_n = a * a_1 + (n * (n - 1) * d) / 2
 *     = n * (a_1 + a_n) / 2
*/

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    long long sum = 0;
    sum = (1 + 20230408LL) * 20230408LL / 2;
    cout << sum;
    return 0;
}
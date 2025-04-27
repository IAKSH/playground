/**
 * https://www.lanqiao.cn/problems/2080/learning/?page=1&first_category_id=1&second_category_id=3&tags=2022
 * 使用乘法分配律配合前缀和减少计算次数
*/

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    int n;
    cin >> n;
    vector<int> v(n);
    vector<long long> prefix(n);
    for(int i = 0;i < n;i++) {
        cin >> v[i];
        prefix[i] = v[i] + (i > 0 ? prefix[i - 1] : 0);
    }

    long long sum = 0;
    for(int i = 0;i < n;i++) {
        sum += v[i] * (prefix[n - 1] - prefix[i]);
    }

    cout << sum << '\n';
    return 0;
}
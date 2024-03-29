/**
 * https://www.lanqiao.cn/problems/2080/learning/?page=1&first_category_id=1&second_category_id=3&tags=2022
 * 没有任何优化的无脑暴力
 * 四例超时
*/

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    int n;
    cin >> n;
    vector<int> v(n);
    for(auto& i : v)
        cin >> i;

    long long sum = 0;
    for(int i = 0;i < n;i++) {
        for(int j = i + 1;j < n;j++) {
            sum += v[i] * v[j];
        }
    }

    cout << sum << '\n';
    return 0;
}
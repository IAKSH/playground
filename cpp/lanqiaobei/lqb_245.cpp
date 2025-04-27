// https://www.lanqiao.cn/problems/245/learning/?page=1&first_category_id=1&tags=%E5%9B%BD%E8%B5%9B,%E9%80%92%E6%8E%A8&sort=pass_rate&asc=0
// 同余，以及找规律（？
// 100%

#include <bits/stdc++.h>

using namespace std;

int main() {
    ios::sync_with_stdio(false);

    int n;
    cin >> n;

    long long res = 0;
    for(int i = 0;i < n;i++) {
        res += static_cast<long long>(i) * (n - i) * (n - i);
        res %= 1000000007;
    }

    cout << res << '\n';
    return 0;
}
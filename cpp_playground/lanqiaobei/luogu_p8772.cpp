// https://www.luogu.com.cn/problem/P8772
// 很简单的拆多项式然后前缀和
// 就是会把int撑爆，以后还是多用long long吧...一定要看各项输入的取值范围
// AC

#include <bits/stdc++.h>

using namespace std;

int main() {
    ios::sync_with_stdio(false);

    int n;cin >> n;
    array<int,200000> a;// real len = n
    for(int i = 0;i < n;i++)
        cin >> a[i];

    array<long long,200000> prefix;// real len = n - 1
    long long acc = 0;
    for(int i = n - 1;i > 0;i--)
        prefix[i - 1] = (acc += a[i]);

    long long res = 0;
    for(int i = 0;i < n - 1;i++) {
        res += a[i] * prefix[i];
    }

    cout << res << '\n';
    return 0;
}
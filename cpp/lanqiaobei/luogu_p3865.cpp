// https://www.luogu.com.cn/problem/P3865
// 先来个无脑的
// 甚至捞了1个AC，2 WA，10 TLE

// https://www.luogu.com.cn/article/61z2lumk
// 正解是用ST表

#include <bits/stdc++.h>

using namespace std;

int main() {
    ios::sync_with_stdio(false);

    int n,m,i,a,b;
    cin >> n >> m;
    int arr[n];
    for(i = 0;i < n;i++)
        cin >> arr[i];

    //int res[m];

    for(i = 0;i < m;i++) {
        cin >> a >> b;
        cout << *max_element(arr + a - 1,arr + b - 1) << '\n';
        //res[i] = *max_element(arr + a - 1,arr + b - 1);
    }

    //for(const auto& i : res)
    //    cout << i << '\n';
    return 0;
}
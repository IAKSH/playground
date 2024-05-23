// https://www.lanqiao.cn/problems/2201/learning/?page=1&first_category_id=1&second_category_id=3&tags=01%E8%83%8C%E5%8C%85
// 倒着来一维DP，带一点点贪心
// 其实原本应该是标准的二维DP，类似01背包，但是实际上可以压缩到一维
// 70%
// 2RE 1WA
// 看了下和别人题解的区别，可能是数组越界了，大概是dp[0]

#include <bits/stdc++.h>

using namespace std;

int main() {
    ios::sync_with_stdio(false);

    int n,cap = 0;
    cin >> n;

    array<array<int,2>,10001> a;// weight,value
    for(int i = 0;i < n;i++) {
        cin >> a[i][0] >> a[i][1];
        cap += a[i][0];
    }

    sort(a.begin(),a.begin() + n,[](const array<int,2>& i,const array<int,2>& j){return (i[0] + i[1]) < (j[0] + j[1]);});

    vector<int> dp(cap);
    for(int i = 0;i < n;i++) {
        for(int j = cap;j >= a[i][0];--j) {
            if(j - a[i][0] <= a[i][1])
                dp[j] = max(dp[j - a[i][0]] + a[i][1],dp[j]);
        }
    }

    cout << *max_element(dp.begin(),dp.end()) << '\n';
    return 0;
}

/*
5
4 4
1 1
5 2
5 5
4 3

10
*/
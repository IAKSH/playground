// https://www.dotcpp.com/oj/problem3152.html

// 疑似 O(N^2) 非线性DP
// 未验证

#include <bits/stdc++.h>

using namespace std;

bool check(int n,int m) noexcept {
    while(m >= 10) {
        m /= 10;
    }
    return m == n % 10;
}

int main() noexcept {
    int n;
    scanf("%d",&n);
    vector<int> v(n);
    for(int i = 0;i < n;i++) {
        int input;
        scanf("%d",&input);
        v[i] = input;
    }

    vector<int> dp(n + 1);
    for(int i = 1;i <= n;i++) {
        auto minn_it = min_element(dp.begin(),dp.begin() + i);
        dp[i] = i - (minn_it - dp.begin()) + *minn_it;
        if(check(v[i - 1],v[i])) {
            ++dp[i];
        }
    }

    cout << n - dp[n] + 1 << '\n';
    return 0;
}
/**
 * https://www.lanqiao.cn/problems/3545/learning/?page=4&first_category_id=1&tags=2023
 * 似乎其实不用dp
 * 好吧，用dp以后应该能优化到O(n)
*/

#include <bits/stdc++.h>

using namespace std;

#ifdef OLD_VER
// bad
int main() noexcept {
    int n;cin >> n;
    vector<char> x(n);
    vector<char> y(n);
    for(auto& c : x) cin >> c;
    for(auto& c : y) cin >> c;

    int cnt = 0;
    for(int i = n - 1;i >= 0;i--) {
        int a = abs(x[i] - y[i]);
        int b = abs(10 + y[i] - x[i]);
        if(a <= b)
            cnt += a;
        else {
            cnt += b;
            for(int j = i - 1;j >= 0;j--) {
                if(y[i - 1] > '0') {
                    --y[i - 1];
                    break;
                }
                y[i - 1] = '9';
            }
        }
    }
    
    cout << cnt << '\n';
    return 0;
}
#else
int main() noexcept {
    int n;cin >> n;
    vector<int> x(n);
    vector<int> y(n);
    char c;
    for(int i = n - 1;i >= 0;i--) {
        cin >> c;
        x[i] = c - '0';
    }
    for(int i = n - 1;i >= 0;i--) {
        cin >> c;
        y[i] = c - '0';
    }

    vector<pair<int,int>> dp(n);
    dp[0].first = (x[0] - y[0] + 10) % 10;  // first = x - y
    dp[0].second = (y[0] - x[0] + 10) % 10; // second = y - x

    for(int i = 1;i < n;i++) {
        dp[i].first = min(
            (x[i] - (x[i - 1] < y[i - 1]) - y[i] + 10) % 10 + dp[i - 1].first,  // 上一次是考虑x退位的情况，下同
            (x[i] + (y[i - 1] < x[i - 1]) - y[i] + 10) % 10 + dp[i - 1].second  // 上一次是考虑y退位的情况，下同
        );
        dp[i].second = min(
            (y[i] - x[i] + (x[i - 1] < y[i - 1]) + 10) % 10 + dp[i - 1].first,
            (y[i] - x[i] - (y[i - 1] < x[i - 1]) + 10) % 10 + dp[i - 1].second
        );
    }

    auto& res = dp[n - 1];
    cout << min(res.first,res.second) << '\n';
    return 0;
}
#endif
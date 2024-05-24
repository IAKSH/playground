//https://www.lanqiao.cn/problems/2186/learning/?page=1&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B,2022&sort=pass_rate&asc=0
// 抄了别人的背包DP实现

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {   
    vector<vector<vector<long long>>> dp(2023,vector<vector<long long>>(11,vector<long long>(2023)));

    for(int i = 0;i <= 2022;i++)
        dp[i][0][0] = 1;// 放不下，什么都不选也是一种选择

    for(int i = 1;i <= 2022;i++) {
        for(int j = 1;j <= 10;j++) {
            for(int k = 1;k <= 2022;k++) {
                dp[i][j][k] = dp[i - 1][j][k];
                if(k >= i)
                    dp[i][j][k] += dp[i - 1][j - 1][k - i];
            }
        }
    }

    cout << dp[2022][10][2022] << '\n';
    return 0;
}
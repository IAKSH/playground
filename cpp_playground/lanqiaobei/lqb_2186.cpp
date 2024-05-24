//https://www.lanqiao.cn/problems/2186/learning/?page=1&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B,2022&sort=pass_rate&asc=0
// 好像做成01背包了，而且没有去重
// 暂存

#include <bits/stdc++.h>

using namespace std;

// 2022拆成10个不同的正整数之和，有多少种方法

int main() noexcept {
    int n = 2022;
    
    //array<array<int,2023>,2023> dp;
    vector<vector<int>> dp(2023,vector<int>(2023));
    for(int i = 0;i <= 2022;i++) {
        for(int j = 0;j <= 2022;j++) {
            // 需要在这里去重，或者在这个循环结束后
            if(i >= j)
                dp[i][j] = dp[i - j][j - 1] + 1;
            else
                dp[i][j] = dp[i][j - 1];
        }
        // 或者在这里去重
    }

    cout << *max_element(dp[2022].begin(),dp[2022].end()) << '\n';
    return 0;
}
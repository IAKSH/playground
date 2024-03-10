// https://leetcode.cn/problems/coin-change

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    int coinChange(vector<int>& coins, int amount) noexcept {
        vector<int> dp(amount + 1,0);
        for(int i = 1;i <= amount;i++) {
            int minn = INT_MAX;
            for(const auto& j : coins) {
                if(i - j >= 0) {
                    minn = min(minn,dp[i - j]);
                }
            }
            // to avoid int overflow
            dp[i] = (minn == INT_MAX) ? INT_MAX : 1 + minn;
        }
        return dp[amount] == INT_MAX ? -1 : dp[amount];
    }
};

int main() noexcept {
    vector<int> v{2};
    Solution s;
    cout << s.coinChange(v,3) << endl;
    return 0;
}
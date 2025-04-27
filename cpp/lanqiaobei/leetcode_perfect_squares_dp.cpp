// https://leetcode.cn/problems/perfect-squares/

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    int numSquares(int n) noexcept {
        vector<int> dp(n + 1,0);
        for(int i = 1;i <= n;i++) {
            int minn = INT_MAX;
            for(int j = 1;j * j <= i;j++) {
                minn = min(minn,dp[i - j * j]);
            }
            dp[i] = 1 + minn;
        }
        return dp[n];
    }
};

int main() noexcept {
    Solution s;
    cout << s.numSquares(13) << endl;
    return 0;
}
// https://leetcode.cn/problems/minimum-path-sum/description/?envType=study-plan-v2&envId=top-100-liked

// 设dp[i][j]为走到对应位置的最小长度
// 由于每次只能向右或下走一步，所以dp[i][j]应该从左和上选择最小的一个dp加map[i][j]的值

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        for(int i = 0;i < grid.size();i++) {
            for(int j = (i == 0 ? 1 : 0);j < grid[0].size();j++) {
                int up = (i == 0 ? INT_MAX : grid[i - 1][j]);
                int left = (j == 0 ? INT_MAX : grid[i][j - 1]);
                grid[i][j] += min(up,left);
            }
        }
        return grid.back().back();
    }
};

int main() {
    vector<vector<int>> grid{{1,3,1},{1,5,1},{4,2,1}};
    cout << Solution().minPathSum(grid) << '\n';
    return 0;
}
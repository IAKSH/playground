// https://leetcode.cn/problems/partition-equal-subset-sum/description/

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int sum = accumulate(nums.begin(),nums.end(),0);
        if(sum & 1 == 1)
            return false;
        int half = sum / 2;

        sort(nums.begin(),nums.end());

        vector<vector<int>> dp(nums.size() + 1,vector<int>(half + 1,0));
        for(int i = 1;i <= nums.size();i++) {
            for(int j = 1;j <= half;j++) {
                int new_val;
                if(nums[i - 1] <= j) {
                    new_val = nums[i - 1];
                    new_val += dp[i - 1][j - nums[i - 1]];
                }
                dp[i][j] = max(new_val,dp[i - 1][j]);
                if(dp[i][j] == half)
                    return true;
            }
        }
        
        return false;
    }
};

int main() noexcept {
    vector<int> v{14,9,8,4,3,2};
    cout << Solution().canPartition(v) << endl;
    return 0;
}
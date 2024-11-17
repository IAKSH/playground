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

        vector<vector<bool>> dp(nums.size() + 1,vector<bool>(half + 1,false));
        for(int i = 1;i <= nums.size();i++) {
            for(int j = 1;j <= half;j++) {
                if(dp[i - 1][j] || nums[i - 1] == j || (j - nums[i - 1] >= 0 && dp[i - 1][j - nums[i - 1]]))
                    dp[i][j] = true;
            }
        }
        
        return dp.back().back();
    }
};

int main() noexcept {
    vector<int> v{14,9,8,4,3,2};
    cout << Solution().canPartition(v) << endl;
    return 0;
}
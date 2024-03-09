// https://leetcode.cn/problems/house-robber

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    int rob(vector<int>& nums) noexcept {
        // dp
        size_t len = nums.size();
        if(len == 0) {
            return 0;
        }
        if(len == 1) {
            return nums[0];
        }

        vector<int> dp(len);
        dp[0] = nums[0];
        dp[1] = max(nums[0],nums[1]);
        for(size_t i = 2;i < len;i++) {
            dp[i] = max(dp[i - 2] + nums[i],dp[i - 1]);
        }
        return *max_element(dp.begin(),dp.end());
    }
};

int main() noexcept {
    vector<int> v{2,7,9,3,1};
    Solution s;
    cout << s.rob(v) << endl;
    return 0;
}
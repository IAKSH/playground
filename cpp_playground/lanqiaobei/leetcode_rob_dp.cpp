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

    int rob1(vector<int>& nums) noexcept {
        // dp + 滚动数组(优化空间复杂度)
        size_t len = nums.size();
        if(len == 0) {
            return 0;
        }
        if(len == 1) {
            return nums[0];
        }

        int first = nums[0];
        int second = max(nums[0],nums[1]);
        int last;

        if(len == 2) {
            return second;
        }
        
        for(size_t i = 2;i < len;i++) {
            last = max(first + nums[i],second);
            first = second;
            second = last;
        }
        return last;
    }
};

int main() noexcept {
    vector<int> v{0,0};
    Solution s;
    cout << s.rob1(v) << endl;
    return 0;
}
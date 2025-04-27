// https://leetcode.cn/problems/longest-increasing-subsequence/

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int len = nums.size();
        vector<int> dp(len + 1);

        for(int i = 0;i < len;i++) {
            int maxn = INT_MIN;
            for(int j = 0;j < i;j++) {
                if(nums[i] > nums[j] && dp[j] > maxn) {
                    maxn = dp[j]; 
                }
            }
            dp[i] = max(1 + maxn,1);
        }

        return *max_element(dp.begin(),dp.end());
    }
};

int main() noexcept {
    vector<int> v{10,9,2,5,3,7,101,18};
    Solution s;
    cout << s.lengthOfLIS(v) << endl;
    return 0;
}

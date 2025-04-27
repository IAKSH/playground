// https://leetcode.cn/problems/subarray-sum-equals-k/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        vector<int> prefix(nums.size() + 1,0);
        int acc = 0,cnt = 0;
        for(int i = 1;i <= nums.size();i++)
            prefix[i] = (acc += nums[i - 1]);

        for(int i = 0;i < prefix.size();i++) {
            for(int j = i + 1;j < prefix.size();j++) {
                if(prefix[j] - prefix[i] == k)
                    ++cnt;
            }
        }

        return cnt;
    }
};

int main() {
    vector<int> nums{1,2,3};
    cout << Solution().subarraySum(nums,3) << '\n';
    return 0;
}
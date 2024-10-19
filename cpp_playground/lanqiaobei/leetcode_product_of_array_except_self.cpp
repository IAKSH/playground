// https://leetcode.cn/problems/product-of-array-except-self/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        vector<int> results;
        int mul;
        for(int i = 0;i < nums.size();i++) {
            mul = 1;
            for(int j = 0;j < nums.size();j++) {
                if(i == j)
                    continue;
                mul *= nums[j];
            }
            results.emplace_back(mul);
        }
        return results;
    }
};

int main() {
    vector<int> nums{1,2,3,4};
    for(const auto& i : Solution().productExceptSelf(nums))
        cout << i << ',';
    cout << "\b \n";
}

/*
输入: nums = [1,2,3,4]
输出: [24,12,8,6]
*/
// https://leetcode.cn/problems/product-of-array-except-self/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        vector<int> results;
        array<vector<int>,2> arr{
            vector<int>(nums.size()),
            vector<int>(nums.size())
        };

        int mul = 1;
        for(int i = 0;i < nums.size();i++)
            arr[0][i] = (mul *= nums[i]);
        mul = 1;
        for(int i = nums.size() - 1;i >= 0;i--)
            arr[1][i] = (mul *= nums[i]);
        
        for(int i = 0;i < nums.size();i++)
            results.emplace_back((i - 1 >= 0 ? arr[0][i - 1] : 1) * (i + 1 < arr[1].size() ? arr[1][i + 1] : 1));

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
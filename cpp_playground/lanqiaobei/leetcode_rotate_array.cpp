// https://leetcode.cn/problems/rotate-array/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        vector<int> res(nums.size());
        for(int i = 0;i < nums.size();i++) {
            res[(i + k) % nums.size()] = nums[i]; 
        }
        nums = res;
    }
};

int main() {
    vector<int> nums{1,2,3,4,5,6,7};
    Solution().rotate(nums,3);
    cout << '{';
    for(const auto& i : nums) {
        cout << i << ',';
    }
    cout << "\b}\n";
}

/*
输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
*/
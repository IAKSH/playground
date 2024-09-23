// https://leetcode.cn/problems/kth-largest-element-in-an-array/description/?envType=study-plan-v2&envId=top-100-liked
// 虽然并不是 O(n)

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        sort(nums.begin(),nums.end(),greater<int>());
        return nums[k - 1];
    }
};

int main() {
    vector<int> nums{3,2,1,5,6,4};
    std::cout << Solution().findKthLargest(nums,2) << '\n';
    return 0;
}
// https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int l = lower_bound(nums.begin(),nums.end(),target) - nums.begin();
        int r = upper_bound(nums.begin(),nums.end(),target) - nums.begin() - 1;
        if(l > r)
            return {-1,-1};
        return {l,r};
    }
};

int main() {
    vector<int> nums{1};
    auto&& v = Solution().searchRange(nums,1);
    cout << '{' << v[0] << ',' << v[1] << "}\n";
    return 0;
}
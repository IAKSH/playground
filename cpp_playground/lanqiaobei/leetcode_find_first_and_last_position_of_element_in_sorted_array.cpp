// https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

//#define USE_STD_BOUND

class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        if(nums.size() == 0)
            return {-1,-1};
#ifdef USE_STD_BOUND
        int r, l = lower_bound(nums.begin(),nums.end(),target) - nums.begin();
        //int r = upper_bound(nums.begin(),nums.end(),target) - nums.begin() - 1;
        for(r = l;r + 1 < nums.size() && nums[r + 1] == target;r++);
        if(l > r)
            return {-1,-1};
        return {l,r};
#else
        int l = 0, r = nums.size() - 1, m = 0;
        while(l < r) {
            m = (l + r) / 2;
            if(nums[m] == target) {
                l = r = m;
                break;
            }
            else if(nums[m] < target) {
                l = m + 1;
            }
            else {
                r = m;
            }
        }

        for(;l - 1 >= 0 && nums[l - 1] == target;l--);
        for(r = l;r + 1 < nums.size() && nums[r + 1] == target;r++);
   
        if(l == r) {
            if(nums[l] == target)
                return {l,l};
            else
                return {-1,-1};
        }
        return {l,r};
#endif
    }
};

int main() {
    vector<int> nums{1,1,2};
    auto&& v = Solution().searchRange(nums,1);
    cout << '{' << v[0] << ',' << v[1] << "}\n";
    return 0;
}
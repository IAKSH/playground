// https://leetcode.cn/problems/search-insert-position/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

#define STD_BINARY_SEARCH

using namespace std;

class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
#ifdef STD_BINARY_SEARCH
        auto it = lower_bound(nums.begin(),nums.end(),target);
        if(it >= nums.end()) {
            nums.emplace_back(target);
            it = nums.end() - 1;
        }
        return it - nums.begin();
#else
        int len = nums.size();
        int l = 0;
        int r = len;
        while(l != r) {
            int mid = (l + r) / 2;
            int val = nums[mid];
            if(val > target)
                r = mid;
            else if(val < target)
                l = mid + 1;
            else {
                l = mid;
                break;
            }
        }
        if(l > len)
            nums.emplace_back(target);
        return l;
#endif
    }
};

int main() noexcept {
    Solution s;
    vector<int> v{1,3,5,6};
    cout << s.searchInsert(v,7) << '\n';
    return 0;
}
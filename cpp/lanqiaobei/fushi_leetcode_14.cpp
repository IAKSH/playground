// https://leetcode.cn/problems/search-insert-position/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int l = 0,r = nums.size(),mid;
        while(l + 1 < r) {
            mid = (l + r) / 2;
            if(nums[mid] == target) {
                return mid;
            }
            else if(nums[mid] > target) {
                r = mid;
            }
            else {
                l = mid + 1;
            }
        }
        if(l >= nums.size() || nums[l] == target)
            return l;
        return nums[l] > target ? l : l + 1;
    }
};

int main() {
    vector<int> v{1,3};
    cout << Solution().searchInsert(v,4) << endl;
    return 0; 
}
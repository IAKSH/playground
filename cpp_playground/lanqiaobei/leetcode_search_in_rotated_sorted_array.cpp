// https://leetcode.cn/problems/search-in-rotated-sorted-array/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    int search(vector<int>& nums, int target) {
        int l = 0, r = nums.size() - 1, m = 0;
        if(nums[l] == nums[r])
            return (nums[0] == target ? 0 : -1);
        if(nums[l] < nums[r])
            return bin_search(nums,target,0,nums.size());
        
        while(l < r) {
            if(nums[l] > nums[r]) {
                m = (l + r) / 2;
                if(nums[m] == target)
                    return m;
                if(nums[m] < nums[l]) {
                    if(m + 1 < nums.size() && nums[m + 1] < nums[m]) {
                        ++m;
                        break;
                    }
                    if(m > 0 && nums[m - 1] > nums[m]) {
                        --m;
                        break;
                    }
                    r = m;
                }
                else {
                    if(m + 1 < nums.size() && nums[m + 1] < nums[m]) {
                        ++m;
                        break;
                    }
                    if(m > 0 && nums[m - 1] > nums[m]) {
                        --m;
                        break;
                    }
                    l = m + 1;
                }
            }
        }
        
        int result_l = bin_search(nums,target,0,m);
        int result_r = bin_search(nums,target,m,nums.size());
        return (result_l == -1 ? result_r : result_l);
    }

private:
    int bin_search(vector<int>& nums,int target,int l ,int r) {
        while(l < r) {
            int m = (r + l) / 2;
            if(nums[m] == target)
                return m;
            else if(nums[m] < target)
                l = m + 1;
            else
                r = m;
        }
        return (l < nums.size() && nums[l] == target ? l : -1);
    }
};

int main() {
    vector<int> nums{1,3};
    cout << Solution().search(nums,4) << '\n';
    return 0;
}

/*
输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4
*/
// https://leetcode.cn/problems/search-in-rotated-sorted-array/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    int search(vector<int>& nums, int target) {
        if(nums.size() == 1)
            return (nums[0] == target ? 0 : -1);
        return locate_broke_point(nums,target,0,nums.size() - 1);
    }

private:
    int locate_broke_point(vector<int>& nums,int target,int l,int r) {
        if(r - l > 1) {
            int m = (l + r) / 2;
            if(nums[m] == target) {
                return m;
            }
            if(nums[m] > nums[m + 1]) {
                // find
                return search_both_side(nums,target,m);
            }
            else {
                int a = locate_broke_point(nums,target,0,m - 1);
                int b = locate_broke_point(nums,target,m + 1,r);
                return (a == -1 ? b : a);
            }
        }
        else {
            for(;l <= r;l++) {
                if(nums[l] == target)
                    return l;
            }
            return -1;
        }
    }

    int search_both_side(vector<int>& nums, int target,int m) {
        int result_l = bin_search(nums,target,0,m);
        int result_r = bin_search(nums,target,m + 1,nums.size() - 1);
        if(result_l != -1)
            return result_l;
        else
            return result_r;
    }

    int bin_search(vector<int>& nums,int target,int l ,int r) {
        //cout << "request bin search in (" << l << ',' << r << "}\n";
        while(l <= r) {
            if(l == r)
                return (nums[l] == target ? l : -1);
            int m = (r + l) / 2;
            if(nums[m] == target)
                return m;
            else if(nums[m] < target)
                l = m + 1;
            else
                r = m;
        }
        return -1;
    }
};

int main() {
    vector<int> nums{1,3,5};
    std::cout << Solution().search(nums,5) << '\n';
    return 0;
}

/*
输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4
*/
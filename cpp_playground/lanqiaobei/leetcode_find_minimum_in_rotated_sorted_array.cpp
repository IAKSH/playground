// https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/description/?envType=study-plan-v2&envId=top-100-liked
// 所谓翻转多次，由公式可得每次只会反转最后一个值到最前
// 也就是说实际上和之前题中的在某一个位置前后对调没有区别
// 用二分找到分离点，然后直接返回前后两个有序区间的第一个值中最小的一个

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    int findMin(vector<int>& nums) {
        int l = 0, r = nums.size() - 1, m = 0;
        if(nums[l] <= nums[r])
            return nums[0];
        
        while(l < r) {
            if(nums[l] > nums[r]) {
                m = (l + r) / 2;
                if(nums[m] < nums[l]) {
                    if(m + 1 < nums.size() && nums[m + 1] < nums[m]) {
                        ++m;
                        break;
                    }
                    if(m > 0 && nums[m - 1] > nums[m]) {
                        //--m;
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
                        //--m;
                        break;
                    }
                    l = m + 1;
                }
            }
        }
        
        int result_l = nums[0];
        int result_r = nums[m];
        return min(result_l,result_r);
    }
};

int main() {
    vector<int> nums{3,4,5,1,2};
    cout << Solution().findMin(nums) << '\n';
    return 0;
}

/*
输入：nums = [3,4,5,1,2]
输出：1
解释：原数组为 [1,2,3,4,5] ，旋转 3 次得到输入数组。
*/
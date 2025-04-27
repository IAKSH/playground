// https://leetcode.cn/problems/maximum-product-subarray/description/

#include <bits/stdc++.h>

using namespace std;

/**
 * can't pass when nums = [2,-5,-2,-4,3]
 * the reason is that we don't know when or where to discard the old dp_negative
 * a flip boolean by itself can't solve this problem
*/
#ifdef OLD_VERSION
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int len = nums.size();
        if(len == 1) {
            return nums[0];
        }

        int maxn = INT_MIN;
        int dp_positive = nums[0];
        int dp_negative = nums[0];
        bool flip = nums[0] > 0;

        for(int i = 1;i < len;i++) {
            dp_positive = max(dp_positive * nums[i],nums[i]);
            dp_negative = min(-(abs(dp_negative) * abs(nums[i])),nums[i]);
            
            if(nums[i] <= 0) {
                flip = !flip;
            }

            maxn = max(maxn,flip ? max(dp_positive,-dp_negative) : dp_positive);
        }

        return maxn;
    }
};
#else
/**
 * [1]:
 * the real question is how to deal with negatives,
 * when there's a negative number, we still want to get the maximum value
 * so we have this:
 * f(i) = f_min(i) * a_i
 * as a_i is negative, obviously f(i) gets bigger when f_min(i) drops down
 * 
 * thus, we need both dp_positive (as f(i)) and dp_negative (as f_min(i))
 * 
 * [2]:
 * multiplying with a negative value will surely flip the value to the other side of zero
 * in that case, the former maximum value will turn to be a new minimum value, vice versa
 * thus we need to swap the max and the min before the next dp_positive and dp_negative computing
*/
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int len = nums.size();
        if(len == 1) {
            return nums[0];
        }

        int maxn = nums[0];
        int dp_positive = nums[0];
        int dp_negative = nums[0];

        for(int i = 1;i < len;i++) {
            // see [2]
            if(nums[i] < 0) {
                swap(dp_positive,dp_negative);
            }
            
            // see [1]
            dp_positive = max(dp_positive * nums[i],nums[i]);
            dp_negative = min(dp_negative * nums[i],nums[i]);

            maxn = max(maxn,dp_positive);
        }

        return maxn;
    }
};
#endif

int main() noexcept {
    vector<int> v{2,-1,1,1};
    Solution s;
    cout << s.maxProduct(v) << endl;
    return 0;
}
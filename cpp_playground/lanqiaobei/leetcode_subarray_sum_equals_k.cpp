// https://leetcode.cn/problems/subarray-sum-equals-k/description/?envType=study-plan-v2&envId=top-100-liked
// TODO: 超时了

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        int len = nums.size(), dl = len, cnt = 0;

        vector<int> prefix(len);
        int acc = 0;
        for(int i = 0;i < len;i++) {
            prefix[i] = (acc += nums[i]);
        }

        while(dl-- > 0) {
            for(int i = 0;i < len - dl;i++) {
                //cnt += (accumulate(nums.begin() + i,nums.begin() + i + dl,0) == k);
                int sum = prefix[i + dl] - (i > 0 ? prefix[i - 1] : 0);
                if(sum == k)
                    ++cnt;
            }
        }
        return cnt;
    }
};

int main() {
    vector<int> nums{1};
    cout << Solution().subarraySum(nums,1) << '\n';
    return 0;
}
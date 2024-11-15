// https://leetcode.cn/problems/partition-equal-subset-sum/description/

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    bool canPartition(vector<int>& nums) {
        half = accumulate(nums.begin(),nums.end(),0);
        half /= 2.0f;
        if(floorf(half) != half)
            return false;
        vector<bool> mask(nums.size(),false);
        for(int i = 0;i < nums.size();i++) {
            if(dfs(nums,mask,0,0))
                return true;
        }
        return false;
    }

private:
    bool dfs(vector<int>& nums,vector<bool>& mask,int acc,int index) {
        mask[index] = true;
        acc += nums[index];
        if(acc == half)
            return true;
        for(int i = 0;i < mask.size();i++) {
            if(!mask[i]) {
                if(dfs(nums,mask,acc,i))
                    return true;
             }
        }
        mask[index] = false;
        return false;
    }

    float half = 0.0f;
};

int main() noexcept {
    vector<int> v{14,9,8,4,3,2};
    cout << Solution().canPartition(v) << endl;
    return 0;
}
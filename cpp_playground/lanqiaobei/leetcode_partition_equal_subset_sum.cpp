// https://leetcode.cn/problems/partition-equal-subset-sum/description/

#include <bits/stdc++.h>

using namespace std;

/**
 *
*/
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int sum = accumulate(nums.begin(),nums.end(),0);
        if(sum % 2 == 1) {
            return false;
        }

        
    }
};

int main() noexcept {
    vector<int> v{1,5,11,5};
    Solution s;
    cout << s.canPartition(v) << endl;
    return 0;
}
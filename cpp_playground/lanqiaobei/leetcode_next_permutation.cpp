// https://leetcode.cn/problems/next-permutation/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        next_permutation(nums.begin(),nums.end());
    }
};

int main() {
    vector<int> nums{3,2,1};
    Solution().nextPermutation(nums);
    for(const auto& i : nums)
        cout << i << ',';
    cout << "\b \n";
    return 0;
}
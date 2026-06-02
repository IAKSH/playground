// https://leetcode.cn/problems/single-number/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>
using namespace std;

class Solution1 {
public:
    int singleNumber(vector<int>& nums) {
        int mask = 0;
        for(int i = 0;i < nums.size();i++) {
            mask ^= nums[i];
        }
        return mask;
    }
};

class Solution {
public:
    int singleNumber(vector<int>& nums) {
        return std::accumulate(std::begin(nums),std::end(nums),0,std::bit_xor<int>());
    }
};

int main() {
    vector<int> v{2,2,4,5,5,6,6};
    cout << Solution().singleNumber(v) << endl;
    return 0;
}
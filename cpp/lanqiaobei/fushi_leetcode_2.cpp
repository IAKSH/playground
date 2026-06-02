// https://leetcode.cn/problems/move-zeroes/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>
using namespace std;

class SolutionUsingSTL {
public:
    void moveZeroes(vector<int>& nums) {
        auto ite = remove(nums.begin(),nums.end(),0);
        while(ite != nums.end())
            *(ite++) = 0;
    }
};

class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        auto first_ite = nums.begin();
        while(first_ite != nums.end() && *first_ite != 0)
            ++first_ite;
        auto ite = first_ite;
        while(ite != nums.end()) {
            if(*ite != 0) {
                *first_ite++ = *ite;
            }
            ++ite;
        }
        while(first_ite != nums.end()) {
            *first_ite = 0;
            ++first_ite;
        }
    }
};

int main() {
    vector<int> v{0,1,0,3,12};
    // 1,1,0,3,12
    Solution().moveZeroes(v);
    for(const auto& i : v) {
        cout << i << ' ';
    }
    cout << endl;
    return 0;
}
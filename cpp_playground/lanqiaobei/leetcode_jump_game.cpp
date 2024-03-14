// https://leetcode.cn/problems/jump-game/description/

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    bool canJump(vector<int>& nums) {
        int len = nums.size();
        int maxn = 0;
        for(int i = 0;i < len - 1;i++) {
            if(i <= maxn && i + nums[i] > maxn)
                maxn = i + nums[i];
        }
        return maxn + 1 >= len;
    }
};

int main() noexcept {
    vector<int> v{3,2,1,0,4};
    Solution s;
    cout << s.canJump(v) << '\n';
    return 0;
}
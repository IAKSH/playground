// https://leetcode.cn/problems/jump-game/description/

#include <bits/stdc++.h>

using namespace std;

/**
 * DFS（用于回退） + 一点点贪心
 * 超时了
*/

class Solution {
public:
    bool canJump(vector<int>& nums,int index = 0) {
        if(index >= nums.size() - 1)
            return true;
        for(int i = nums[index];i > 0;i--) {
            if(canJump(nums,index + i))
                return true;
        }
        return false;
    }
};

int main() noexcept {
    vector<int> v{3,2,1,0,4};
    Solution s;
    cout << s.canJump(v) << '\n';
    return 0;
}
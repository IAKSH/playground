// https://leetcode.cn/problems/3sum/description/?envType=study-plan-v2&envId=top-100-liked
// 看起来好像就是暴力+记忆化减枝
// 然后事先排序一下可以省略很多步骤，甚至由于有序了，可以通过单调递增排除很多枝
// 也就是用双指针

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> res;
        sort(nums.begin(),nums.end());
        int i = 0,j,k = nums.size() - 1;
        while(i != k) {
            // 最后一个令sum小于0的值
            int last_j_val = INT_MIN;
            for(j = i + 1;j < k;j++) {
                // 如果还是之前哪个令sum小于0的值就直接跳过
                if(nums[j] == last_j_val)
                    continue;
                int sum = nums[i] + nums[j] + nums[k];
                if(sum == 0) {
                    res.push_back(vector<int>{nums[i],nums[j],nums[k]});
                    // 由于单调递增，后续的j一定会使sum大于0，直接跳过
                    break;
                }
                // 记录之前的会导致sum小于0的值
                last_j_val = nums[j];
            }
            // 现在应该考虑更新i和k了，在j和k相撞之前，自last_j到k之间，全部的j都能使sum大于0
            // k是自最大向最小移动的，所以可以尝试减小k来寻找符合条件的j
            // 而当且仅当j和k相撞时，j没有上升的余量，所有j都只能使sum小于0
            // 而i又是从最小向最大移动的，所以此时考虑令i上升以寻找符合的j
            if(j == k)
                ++i;
            else
                --k;
        }
        
        // 不知道为什么有重复的，暂且用个unique修一下
        int n = res.end() - unique(res.begin(),res.end());
        while(n-- > 0)
            res.pop_back();
        return res;

        // 好吧，还是有问题
    }
};

void test(vector<int>&& nums) {
    cout << "Testing: {";
    for(const auto& i : nums) {
        cout << i << ',';
    }
    cout << "\b}\nResult:\n";
    for(const auto& v : Solution().threeSum(nums)) {
        cout << '{' << v[0] << ',' << v[1] << ',' << v[2] << "}\n";
    }
}

int main() {
    test({0,0,0,0});
    test({-1,0,1,2,-1,-4});
    return 0;
}

/*
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
*/
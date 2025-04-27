// https://leetcode.cn/problems/permutations/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> results;
        dfs(results,nums,0);
        return results;
    }

private:
    void dfs(vector<vector<int>>& results,vector<int>& nums,int depth) {
        if(depth == nums.size() - 1)
            results.emplace_back(nums);
        else {
            for(int i = depth;i < nums.size();i++) {
                swap(nums[depth],nums[i]);
                dfs(results,nums,depth + 1);
                swap(nums[depth],nums[i]);
            }
        }
    }
};

int main() {
    vector<int> nums{1,2,3};
    cout << '{';
    for(const auto& v : Solution().permute(nums)) {
        cout << '{';
        for(const auto& i : v)
            cout << i << ',';
        cout << "\b},";
    };
    cout << "\b}\n";
}
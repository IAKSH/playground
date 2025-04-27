// https://leetcode.cn/problems/combination-sum/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        sort(candidates.begin(),candidates.end());
        vector<vector<int>> results;
        unordered_set<string> mem;
        for(const auto& i : candidates)
            dfs(results,mem,candidates,std::vector<int>(),target);
        return results;
    }

private:
    bool dfs(vector<vector<int>>& results,unordered_set<string>& mem,vector<int>& nums,vector<int> v,int target) {
        int sum = accumulate(v.begin(),v.end(),0);
        if(sum == target) {
            sort(v.begin(),v.end());
            stringstream ss;
            for(const auto& i : v)
                ss << i;
            if(mem.count(ss.str()) == 0) {
                results.emplace_back(v);
                mem.emplace(ss.str());
            }
            return false;
        }
        else if(sum < target) {
            for(const auto& i : nums) {
                if(v.empty() || i >= v.back()) {
                    vector<int> new_v = v;
                    new_v.emplace_back(i);
                    if(!dfs(results,mem,nums,new_v,target))
                        break;
                }
            }
            return true;
        }
        return false;
    }
};

int main() {
    vector<int> ca{2,3,6,7};
    for(const auto& v : Solution().combinationSum(ca,7)) {
        cout << '{';
        for(const auto& i : v)
            cout << i << ',';
        cout << "\b},";
    }
    cout << "\b \n";
    return 0;
}
// https://leetcode.cn/problems/subsets/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> results{{}};
    
        for(const auto& n : nums) {
            vector<vector<int>> buffer;
            for(const auto& v : results) {
                auto new_v = v;
                new_v.emplace_back(n);
                buffer.emplace_back(new_v);
            }
            for(const auto& v : buffer)
                results.emplace_back(v);
        }
        return results;
    }
};

int main() {
    vector<int> nums {1,2,3};
    cout << '{';
    for(const auto& v : Solution().subsets(nums)) {
        cout << '{';
        for(const auto& i : v) {
            cout << i << ',';
        }
        cout << "\b},";
    }
    cout << "\b}\n";
}
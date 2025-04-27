// https://leetcode.cn/problems/3sum/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;


class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<vector<int>> results;
        
        auto l = nums.begin(),r = nums.end();
        if (l >= r)
            return results;
        for (auto i = l; i < r - 2; ++i) {
            // 跳过重复
            if (i > l && *i == *(i - 1)) continue;
            auto j = i + 1;
            auto k = r - 1;
            while (j < k) {
                int sum = *i + *j + *k;
                if (sum < 0) {
                    ++j;
                } else if (sum > 0) {
                    --k;
                } else {
                    results.emplace_back(vector<int>{*i, *j, *k});
                    // 跳过重复
                    while (j < k && *j == *(j + 1)) ++j;
                    while (j < k && *k == *(k - 1)) --k;
                    ++j;
                    --k;
                }
            }
        }

        return results;
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
    //test({0,0,0,0});
    //test({-1,0,1,2,-1,-4});
    test({-1,0,1,2,-1,-4,-2,-3,3,0,4});
    return 0;
}
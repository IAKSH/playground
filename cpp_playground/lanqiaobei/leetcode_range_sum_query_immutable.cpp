// https://leetcode.cn/problems/range-sum-query-immutable/description/

#include <bits/stdc++.h>

using namespace std;

class NumArray {
public:
    NumArray(vector<int>& nums) {
        int len = nums.size(), acc = 0;
        prefix.resize(len + 1);
        for(int i = 0;i < len;i++) {
            prefix[i + 1] = (acc += nums[i]);
        }
    }
    
    int sumRange(int left, int right) {
        return prefix[right + 1] - prefix[left];
    }

private:
    vector<int> prefix;
};

int main() {
    vector<int> nums{-2,0,3,-5,2,-1};
    NumArray na(nums);
    for(const auto& v : vector<pair<int,int>>{{0,2},{2,5},{0,5}}) {
        cout << na.sumRange(v.first,v.second) << '\n';
    }
    return 0;
}

/*
[[[-2, 0, 3, -5, 2, -1]], [0, 2], [2, 5], [0, 5]]
= [null, 1, -1, -3]
*/
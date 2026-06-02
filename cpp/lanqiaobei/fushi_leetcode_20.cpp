// https://leetcode.cn/problems/majority-element/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>
using namespace std;

class Solution1 {
public:
    int majorityElement(vector<int>& nums) {
        unordered_map<int,int> map;
        for(const auto& i : nums) {
            ++map[i];
        }
        auto max_it = map.begin();
        for(auto it = map.begin();it != map.end();it++) {
            if(it->second > max_it->second)
                max_it = it;
        }
        return max_it->first;
    }
};

// 擂台
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int n = nums[0], score = 1;
        for(int i = 1;i < nums.size();i++) {
            if(nums[i] == n)
                ++score;
            else {
                if(score)
                    --score;
                else {
                    n = nums[i];
                    score = 1;
                }
            }
        }
        return n;
    }
};

int main() {

}
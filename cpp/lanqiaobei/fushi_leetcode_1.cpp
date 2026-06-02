// https://leetcode.cn/problems/two-sum/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int,int> map;
        for(int i = 0;i < nums.size();i++) {
            map[nums[i]] = i;
        }
        for(int i = 0;i < nums.size();i++) {
            auto ite = map.find(target - nums[i]);
            if(ite != map.end() && ite->second != i) {
                return {i,ite->second};
            }
        }
        return {0,0};
    }
};

int main() {

}
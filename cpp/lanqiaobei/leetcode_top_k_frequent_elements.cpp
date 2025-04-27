// https://leetcode.cn/problems/top-k-frequent-elements/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        sort(nums.begin(),nums.end());
        vector<pair<int,int>> v{make_pair(nums[0],1)};
        int last_num = nums[0];
        for(int i = 1;i < nums.size();i++) {
            if(nums[i] == last_num)
                ++v.back().second;
            else {
                v.emplace_back(make_pair(nums[i],1));
                last_num = nums[i];
            }
        }
        sort(v.begin(),v.end(),[](const pair<int,int>& p1,const pair<int,int>& p2){
            return p1.second > p2.second;
        });
        vector<int> res(k);
        for(int i = 0;i < k;i++)
            res[i] = v[i].first;
        return res;
    }
};

int main() {
    vector<int> nums{5,2,5,3,5,3,1,1,3};
    for(const auto& i : Solution().topKFrequent(nums,2))
        cout << i << ' ';
    cout << '\n';
    return 0;
}
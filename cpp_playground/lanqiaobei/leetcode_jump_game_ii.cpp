// https://leetcode.cn/problems/jump-game-ii/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    int jump(vector<int>& nums) {
        int len = nums.size();
        vector<pair<int,int>> v(len);
        for(int i = 0;i < len - 1;i++) {
            v[i].first = nums[i];
            v[i].second = i;
        }

        sort(v.begin(),v.end(),[&](const pair<int,int>& p1,const pair<int,int>& p2){
            return (p1.first + (len - 1 - (p1.first + p1.second))) > (p2.first + (len - 1 - (p2.first + p2.second)));
        });

        int n = len - 1,cnt = 0;
        while(!v.empty() && n > 0) {
            for(auto it = v.begin();it != v.end();it++) {
                if(it->first >= n - it->second) {
                    ++cnt;
                    n = it->second;
                    v.erase(it);
                    break;
                }
            }
        }

        return cnt;
    }
};

int main() {
    vector<int> nums{2,0,2,4,6,0,0,3};
    cout << Solution().jump(nums) << '\n';
    return 0;
}

/*
输入: nums = [2,3,1,1,4]
输出: 2
*/
// https://leetcode.cn/problems/sliding-window-maximum/?utm_source=LCUS&utm_medium=ip_redirect&utm_campaign=transfer2china
// 优先队列，不知道为什么还是TLE
// 37 / 51 TLE

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> res;
        deque<pair<int,int>> pq;//val,index

        int i,j,n = nums.size() - k;
        for(i = 0;i <= n;i++) {
            for(j = 0;j < k;j++) {
                while(!pq.empty() && pq.back().first <= nums[i + j])
                    pq.pop_back();
                pq.emplace_back(pair<int,int>(nums[i + j],i + j));
            }
            res.emplace_back(pq.front().first);
            if(i == pq.front().second)
                pq.pop_front();
        }

        return res;
    }
};

int main() {
    //nums = [1,3,-1,-3,5,3,6,7], k = 3; should be 3 3 5 5 6 7
    //nums = [1,3,1,2,0,5],k = 3; should be [3,3,2,5]
    vector<int> v{1,3,-1,-3,5,3,6,7};
    int k = 3;
    for(const auto& i : Solution().maxSlidingWindow(v,k))
        cout << i << ' ';
    cout << '\n';
    return 0;
}
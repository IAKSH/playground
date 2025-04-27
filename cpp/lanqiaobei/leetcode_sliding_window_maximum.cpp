// https://leetcode.cn/problems/sliding-window-maximum/?utm_source=LCUS&utm_medium=ip_redirect&utm_campaign=transfer2china
// O(n)的优先队列
// AC

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> res;
        deque<int> pq;// 实际上不需要考虑重复，因为即便重复也能够像下面这样处理

        int i,n = nums.size();
        for(i = 0;i < n;i++) {
            while(!pq.empty() && pq.back() < nums[i])
                pq.pop_back();
            pq.emplace_back(nums[i]);
            if(i >= k - 1) {
                res.emplace_back(pq.front());
                if(nums[i - k + 1] == pq.front())
                    pq.pop_front();
            }
        }

        return res;
    }
};

int main() {
    //nums = [1,3,-1,-3,5,3,6,7], k = 3; should be 3 3 5 5 6 7
    vector<int> v{1,3,-1,-3,5,3,6,7};

    //nums = [1,3,1,2,0,5],k = 3; should be [3,3,2,5]
    //vector<int> v{1,3,1,2,0,5};
    int k = 3;

    // -7,-8,7,5,7,1,6,0
    // 4
    // should be [7,7,7,7,7]
    //vector<int> v{-7,-8,7,5,7,1,6,0};
    //int k = 4;
    
    for(const auto& i : Solution().maxSlidingWindow(v,k))
        cout << i << ' ';
    cout << '\n';
    return 0;
}
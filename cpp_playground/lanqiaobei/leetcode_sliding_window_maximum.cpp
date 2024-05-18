// https://leetcode.cn/problems/sliding-window-maximum/?utm_source=LCUS&utm_medium=ip_redirect&utm_campaign=transfer2china
// 先写一个无脑的，不出所料超时了
// 37 / 51 TLE

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> res;
        for(auto it = nums.begin();it != nums.end() - k + 1;it++) {
            res.emplace_back(*max_element(it,it + k));
        }
        return res;
    }
};

int main() {
    //nums = [1,3,-1,-3,5,3,6,7], k = 3
    vector<int> v{1,3,-1,-3,5,3,6,7};
    int k = 3;
    for(const auto& i : Solution().maxSlidingWindow(v,k))
        cout << i << ' ';
    cout << '\n';
    return 0;
}
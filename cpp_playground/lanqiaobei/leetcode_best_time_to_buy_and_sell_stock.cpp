// https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/description/

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    int maxProfit(vector<int>& prices) noexcept {
        int minn = INT_MAX;
        int profit = 0;
        for(const auto& price : prices) {
            minn = min(minn,price);
            profit = max(profit,price - minn);
        }
        return profit;
    }
};

int main() noexcept {
    vector<int> v{2,4,1};
    Solution s;
    cout << s.maxProfit(v) << endl;
    return 0;
}
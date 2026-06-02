// https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int ever_min_p = INT_MAX;
        int max_profit = 0;
        for(const auto p : prices) {
            ever_min_p = min(ever_min_p,p);
            max_profit = max(max_profit,p - ever_min_p);
        }
        return max_profit;
    }
};

int main() {

}
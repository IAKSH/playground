// https://leetcode.cn/problems/daily-temperatures/description/?envType=study-plan-v2&envId=top-100-liked
// 单调栈

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        stack<pair<int,int>> s;
        vector<int> result(temperatures.size(),0);
        for(int i = 0;i < temperatures.size();i++) {
            while(!s.empty() && temperatures[i] > s.top().first) {
                result[s.top().second] = i - s.top().second;
                s.pop();
            }
            s.emplace(make_pair(temperatures[i],i));
        }
        return result;
    }
};

int main() {
    vector<int> temps{73,74,75,71,69,72,76,73};
    cout << '{';
    for(const auto& i : Solution().dailyTemperatures(temps))
        cout << i << ',';
    cout << "\b}\n";
}

/*
输入: temperatures = [73,74,75,71,69,72,76,73]
输出: [1,1,4,2,1,1,0,0]
*/
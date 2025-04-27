// https://leetcode.cn/problems/merge-intervals/submissions/567705865/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        vector<vector<int>> res;
        sort(intervals.begin(),intervals.end());
        for(int i = 0;i < intervals.size();) {
            int val = intervals[i][1];
            int j = i + 1;
            while (j < intervals.size() && intervals[j][0] <= val) {
                val = max(val, intervals[j][1]);
                j++;
            }
            res.emplace_back(vector<int>{intervals[i][0],val});
            i = j;
        }
        return res;
    }
};

void test(vector<vector<int>>&& intervals) {
    cout << "\nTest on {";
    for(const auto& v : intervals) {
        cout << '{' << v[0] << ',' << v[1] << '}';
    }
    cout << "}\nResult:\n";
    for(const auto& v : Solution().merge(intervals)) {
        cout << '{' << v[0] << ',' << v[1] << "}\n";
    }
}

int main() {
    test(vector<vector<int>>{{1,2},{2,6},{8,10},{15,18}});
    test(vector<vector<int>>{{1,4},{2,3}});
    test(vector<vector<int>>{{2,3},{4,5},{6,7},{8,9},{1,10}});//{1,10},{2,3},{4,5},{6,7},{8,9}
    test(vector<vector<int>>{{1,4},{0,0}});
    test(vector<vector<int>>{{1,4},{0,2},{3,5}});
    return 0;
}
// https://leetcode.cn/problems/merge-intervals/submissions/567705865/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        sort(intervals.begin(),intervals.end(),[](const vector<int>& a,const vector<int>& b){
            return a[0] < b[0];
        });

        for(auto i = intervals.begin();i != intervals.end();) {
            bool go = true;
            for(auto j = i + 1;j != intervals.end();j++) {
                if(i->back() >= j->front()) {
                    j->front() = i->front();
                    j->back() = max(j->back(),i->back());
                    intervals.erase(i);
                    go = false;
                    break;
                }
            }
            if(go)
                ++i;
        }

        return intervals;
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
    test(vector<vector<int>>{{2,3},{4,5},{6,7},{8,9},{1,10}});
    test(vector<vector<int>>{{1,4},{0,0}});
    return 0;
}
// https://leetcode.cn/problems/course-schedule/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        for(const auto& i : prerequisites) {
            v2.emplace_back(make_pair(i[0],i[1]));
            while(!v2.empty()) {
                if(!insert(v2.front().first,v2.front().second))
                    return false;
                v2.pop_front();
            }
        }
        return true;
    }

private:
    bool insert(int i,int j) {
        if(i == j)
            return false;
        for(const auto& p : v1) {
            if(p.second == i) {
                if(p.first == j)
                    return false;
                v2.emplace_back(make_pair(p.first,j));
            }
        }
        v1.emplace_back(make_pair(i,j));
        return true;
    }

    deque<pair<int,int>> v1,v2;
};

int main() {
    cout << "I don't want to write a test\n";
    return 0;
}
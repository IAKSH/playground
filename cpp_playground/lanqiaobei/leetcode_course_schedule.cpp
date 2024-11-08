// https://leetcode.cn/problems/course-schedule/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        if(prerequisites.size() == 0)
            return true;
        for(const auto& v : prerequisites) {
            if(v[0] == v[1])
                return false;
            mem[v[0]].emplace_back(v[1]);
        }
        for(const auto& v : prerequisites) {
            if(!dfs(v[0]))
                return false;
        }
        return true;
    }

private:
    bool dfs(int x) {
        if(marks.count(x) != 0)
            return false;
        marks.emplace(x);
        for(const auto& i : mem[x]) {
            if(!dfs(i))
                return false;
        }
        marks.erase(x);
        return true;
    }

    unordered_map<int,vector<int>> mem;
    unordered_set<int> marks;
};

int main() {
    cout << "I don't want to write a test\n";
    return 0;
}
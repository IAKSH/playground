// https://leetcode.cn/problems/course-schedule/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        mem.resize(numCourses);
        marks.resize(numCourses, false);
        subtree_mark.resize(numCourses, false);

        for(const auto& v : prerequisites) {
            if(v[0] == v[1])
                return false;
            mem[v[0]].emplace_back(v[1]);
        }

        // 因为课程记为 0 到 numCourses - 1
        for(int i = 0; i < numCourses; ++i) {
            if(!dfs(i))
                return false;
        }

        return true;
    }

private:
    bool dfs(int x) {
        if(marks[x])
            return false;
        if(subtree_mark[x])
            return true;

        marks[x] = true;
        for(const auto& i : mem[x]) {
            if(!dfs(i))
                return false;
        }
        marks[x] = false;
        subtree_mark[x] = true;
        return true;
    }

    vector<vector<int>> mem;
    vector<bool> marks, subtree_mark;
};

int main() {
    cout << "I don't want to write a test\n";
    return 0;
}
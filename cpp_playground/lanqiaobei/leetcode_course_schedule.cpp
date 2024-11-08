// https://leetcode.cn/problems/course-schedule/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        if(prerequisites.size() == 0)
            return true;
        mem.resize(numCourses);
        marks.resize(numCourses);
        subtree_mark.resize(numCourses);

        for(const auto& v : prerequisites) {
            if(v[0] == v[1])
                return false;
            mem[v[0]].emplace_back(v[1]);
        }
        for(const auto& v : prerequisites) {
            if(!dfs(v[0]))
                return false;
            subtree_mark[v[0]] = true;
        }
        return true;
    }

private:
    bool dfs(int x) {
        marks[x] = true;
        for(const auto& i : mem[x]) {
            if(!subtree_mark[i] && (marks[i] || !dfs(i)))
                return false;
        }
        marks[x] = false;
        return true;
    }

    vector<vector<int>> mem;
    vector<bool> marks,subtree_mark;
};

int main() {
    cout << "I don't want to write a test\n";
    return 0;
}
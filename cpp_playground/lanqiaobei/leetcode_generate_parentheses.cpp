// https://leetcode.cn/problems/generate-parentheses/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    vector<string> generateParenthesis(int n) {
        vector<string> results;
        if(n > 0)
            dfs(results,n,"(",1,0);
        return results;
    }

private:
    void dfs(vector<string>& results,int n,string s,int l_cnt,int r_cnt) {
        if(l_cnt > n)
            return;
        else if(l_cnt == n && l_cnt == r_cnt)
            results.emplace_back(s);
        else {
            if(l_cnt > r_cnt)
                dfs(results,n,s + ')',l_cnt,r_cnt + 1);
            dfs(results,n,s + '(',l_cnt + 1,r_cnt);
        }
    }
};

int main() {
    for(const auto& s : Solution().generateParenthesis(3))
        cout << '\"' <<  s << "\",";
    cout << "\b \n";
    return 0;
}
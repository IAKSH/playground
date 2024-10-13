// https://leetcode.cn/problems/palindrome-partitioning/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    vector<vector<string>> partition(string s) {
        vector<vector<string>> results;
        dfs(s,results,vector<string>(),0,s.size());
        return results;
    }

private:
    void dfs(const string& s,vector<vector<string>>& results,vector<string> v,int l,int len) {
        v.emplace_back(s.substr(l,len));
        if(l + len == s.size()) {
            if(check(v.back()))
                results.emplace_back(v);
        }
        for(int i = 1;i < s.size() - l;i++) {
            auto new_v = v;
            new_v.pop_back();
            new_v.emplace_back(s.substr(l,i));
            if(!check(new_v.back()))
                continue;
            dfs(s,results,new_v,l + i,s.size() - l - i);
        }
    }

    bool check(const string& s) {
        if(s.size() == 1)
            return true;
        int l = 0,r = s.size() - 1;
        while(l < r) {
            if(s[l] != s[r])
                return false;
            ++l;
            --r;
        }
        return true;
    }
};

int main() {
    for(const auto& v : Solution().partition("aab")) {
        cout << '{';
        for(const auto& c : v) {
            cout << c << ',';
        }
        cout << "\b},";
    }
    cout << "\b \n";
    return 0;
}
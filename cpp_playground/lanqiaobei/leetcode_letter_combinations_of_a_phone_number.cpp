// https://leetcode.cn/problems/letter-combinations-of-a-phone-number/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    vector<string> letterCombinations(string digits) {
        array<vector<char>,8> chars{
            vector<char>{'a','b','c'},
            vector<char>{'d','e','f'},
            vector<char>{'g','h','i'},
            vector<char>{'j','k','l'},
            vector<char>{'m','n','o'},
            vector<char>{'p','q','r','s'},
            vector<char>{'t','u','v'},
            vector<char>{'w','x','y','z'}
        };

        vector<string> results;
        vector<vector<char>> layers;
        if(digits != "") {
            for(const auto& c : digits)
                layers.emplace_back(chars[c - '0' - 2]);
            for(const auto& c : layers[0])
                dfs(results,layers,string{c},0);
        }
        return results;
    }

private:
    void dfs(vector<string>& results,const vector<vector<char>>& layers,string s, int depth) {
        if(depth == layers.size() - 1)
            results.emplace_back(s);
        else {
            for(const auto& c : layers[depth + 1])
                dfs(results,layers,s + c,depth + 1);
        }
    }
};

int main() {
    for(const auto& i : Solution().letterCombinations("23"))
        cout << i << ',';
    cout << "\b \n";
}

/*
输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
*/
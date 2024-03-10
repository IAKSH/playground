// https://leetcode.cn/problems/word-break

#include <iostream>
#include <algorithm>
#include <vector>

#define USE_HASHSET
//#define DEBUG

#ifdef USE_HASHSET
#include <unordered_set>
#endif

using namespace std;

/**
 * can't pass when:
 * s = "cars"
 * dict = ["car","ca","rs"]
 * 
 * if we insist on this, we might need to do dfs in tree (together with memoization and pruning)
 * that might be able to solve this problem, but will be much more complex
 * better to choose dp
*/
#ifdef NO_DP
class Solution {
public:
    bool wordBreak(string s, vector<string>& dict) noexcept {
        for(const auto& word : dict) {
            string::iterator it;
            while((it = find_end(s.begin(),s.end(),word.begin(),word.end())) != s.end()) {
                fill(it,it + word.size(),' ');
            }
        }

        for(const auto& c : s) {
            if(c != ' ') {
                return false;
            }
        }
        return true;
    }
};
#else
class Solution {
private:
    string* s;
    size_t len;
#ifdef USE_HASHSET
    unordered_set<string> dict;
#else
    vector<string>* dict;
#endif

    bool check(int j,int i) noexcept {
#ifdef DEBUG
        cout << "checking (" << j << ',' << i << ")\t" << s->substr(j,i) << endl;
#endif
#ifdef USE_HASHSET
        return find(dict.begin(),dict.end(),s->substr(j,i)) != dict.end();
#else
        return find(dict->begin(),dict->end(),s->substr(j,i)) != dict->end();
#endif
    }

public:
    bool wordBreak(string s, vector<string>& dict) noexcept {
        this->s = &s;
        this->len = s.size();
#ifdef USE_HASHSET
        for(const auto& v : dict) {
            this->dict.insert(v);
        }
#else
        this->dict = &dict;
#endif
        
        vector<bool> dp(len + 1);
        dp[0] = true;

        for(int i = 1;i <= len;i++) {
            for(int j = 0;j < i;j++) {
                if(dp[j] && check(j,i - j)) {
                    dp[i] = true;
                    break;
                }
            }
        }

        return dp[len];
    }
};
#endif

int main() noexcept {
    string s = "applepenapple";
    vector<string> dict{"apple","pen"};

    Solution solution;
    cout << (solution.wordBreak(s,dict) ? "true" : "false") << endl;
    return 0;
}
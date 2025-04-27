// https://leetcode.cn/problems/longest-substring-without-repeating-characters/submissions/573300495/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        if(s.size() == 0)
            return 0;

        int maxn = 0;
        auto l = s.begin();
        for(auto it = l + 1;true;it++) {
            maxn = max(maxn,static_cast<int>(it - l));
            if(it == s.end())
                break;
            while(l != it && check(l,it))
                ++l;
        }
        return maxn;
    }

private:
    bool check(string::iterator begin,string::iterator end) {
        while(begin != end) {
            if(*begin++ == *end)
                return true;
        }
        return false;
    }
};

int main() {
    cout << Solution().lengthOfLongestSubstring("") << '\n';
    return 0;
}
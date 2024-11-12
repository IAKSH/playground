// https://leetcode.cn/problems/decode-string/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    string decodeString(string s) {
        int l,r;
        for(l = 0;l < s.size();l++) {
            if(s[l] == '[')
                break;
        }
        for(r = s.size() - 1;r > l;r--) {
            if(s[r] == ']')
                break;
        }
        if(l >= r)
            return s;
        else {
            string str = decodeString(s.substr(l + 1,r - l - 1));
            string res(s.substr(0,l));
            if(is_num(res.back()))
                res.pop_back();
            if(l > 0 && is_num(s[l - 1])) {
                for(int i = 0;i < to_num(s[l - 1]);i++)
                    res += str;
            }
            res += s.substr(r + 1,s.size() - r);
            return res;
        }
    }

private:
    bool is_num(char c) {
        return c >= '0' && c <= '9';
    }

    int to_num(char c) {
        return c - '0';
    }
};

int main() {
    cout << Solution().decodeString("abc3[cd]xyz") << '\n';
    return 0;
}

/*
输入：s = "3[a]2[bc]"
输出："aaabcbc"

输入：s = "3[a2[c]]"
输出："accaccacc"

输入：s = "abc3[cd]xyz"
输出："abccdcdcdxyz"

"3[z]2[2[y]pq4[2[jk]e1[f]]]ef"
zzzyypqjkjkefjkjkefjkjkefjkjkefyypqjkjkefjkjkefjkjkefjkjkefef

*/
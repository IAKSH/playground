// https://leetcode.cn/problems/decode-string/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    string decodeString(string s) {
        stack<int> st;
        for(int i = 0;i < s.size();i++) {
            if(s[i] == '[')
                st.emplace(i);
            else if(s[i] == ']') {
                int n = 0;
                int j;
                for(j = st.top() - 1;j >= 0 && s[j] >= '0' && s[j] <= '9';j--)
                    n += pow(10,st.top() - 1 - j) * (s[j] - '0');

                string l(s.substr(0,j + 1));
                string r(s.substr(i + 1));
                string m(s.substr(st.top() + 1,i - st.top() - 1));

                for(int k = 0;k < n;k++)
                    l += m;
                l += r;
                i = st.top() - 1;
                s = l;
                st.pop();
            }
        }
        return s;
    }
};

int main() {
    cout << Solution().decodeString("3[z]2[2[y]pq4[2[jk]e1[f]]]ef") << '\n';
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
zzzyypqjkjkefjkjkefjkjkefjkjkefyypqjkjkefjkjkefjkjkefjkjkefef
zzzyypqjkjkefjkjkefjkjkefjkjkefyypqjkjkefjkjkefjkjkefjkjkefef
zzzyypqjkjkefjkjkefjkjkefjkjkefyypqjkjkefjkjkefjkjkefjkjkefef

"100[leetcode]"

*/
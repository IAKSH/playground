// https://leetcode.cn/problems/decode-string/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    string decodeString(string s) {
        stack<int> st;
        stringstream ss;
        for(int i = 0;i < s.size();i++) {
            if(s[i] == '[')
                st.emplace(i);
            else if(s[i] == ']') {
                int n = 0;
                int j;
                for(j = st.top() - 1;j >= 0 && s[j] >= '0' && s[j] <= '9';j--)
                    n += pow(10,st.top() - 1 - j) * (s[j] - '0');

                ss << s.substr(0,j + 1);
                for(int k = 0;k < n;k++)
                    ss << s.substr(st.top() + 1,i - st.top() - 1);
                ss << s.substr(i + 1);
                s = ss.str();
                ss.str("");

                i = st.top() - 1;
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
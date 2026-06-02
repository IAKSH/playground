// https://leetcode.cn/problems/valid-parentheses/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>
using namespace std;

class Solution1 {
public:
    bool isValid(string s) {
        stack<char> st;
        for(const auto& c : s) {
            if(c == '(' || c == '[' || c == '{')
                st.emplace(c);
            else {
                if(st.empty())
                    return false;

                switch(st.top()) {
                case '(':
                    if(c != ')')
                        return false;
                    st.pop();
                    break;
                case '[':
                    if(c != ']')
                        return false;
                    st.pop();
                    break;
                case '{':
                    if(c != '}')
                        return false;
                    st.pop();
                    break;
                default:
                    return false;
                }
            }
        }
        return st.empty();
    }
};

class Solution {
public:
    bool isValid(string s) {
        if(!s.size())
            return true;
        if(s.size() & 1)
            return false;
        
        stack<char> st;
        for(const auto& c : s) {
            if(c == '(')
                st.emplace(')');
            else if(c == '[')
                st.emplace(']');
            else if(c == '{')
                st.emplace('}');
            else {
                if(st.empty() || st.top() != c)
                    return false;
                st.pop();
            }
        }

        return st.empty();
    }
};

int main() {

}
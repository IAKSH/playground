// https://leetcode.cn/problems/longest-palindromic-substring/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    string longestPalindrome(string s) {
        int max_len = 0,res_l = 0,res_r = 0,len,l,r;
        for(int i = 0;i < s.size();i++) {
            l = r = i;
            len = 1;
            while(l > 0 && s[l - 1] == s[i]) {
                --l;
                ++len;
                if(len > max_len) {
                    max_len = len;
                    res_l = l;
                    res_r = r;
                }
            }
            while(r < s.size() - 1 && s[r + 1] == s[i]) {
                ++r;
                ++len;
                if(len > max_len) {
                    max_len = len;
                    res_l = l;
                    res_r = r;
                }
            }
            while(l > 0 && r < s.size() - 1 && s[l - 1] == s[r + 1]) {
                --l;
                ++r;
                len += 2;
                if(len > max_len) {
                    max_len = len;
                    res_l = l;
                    res_r = r;
                }
            };
        }
        return s.substr(res_l,res_r - res_l + 1);
    }
};

int main() {
    cout << Solution().longestPalindrome("aacabdkacaa") << '\n';
    return 0;
}
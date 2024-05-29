// https://leetcode.cn/problems/longest-common-subsequence/description/
// 中规中矩的二维dp，用二维图标来思考的话就很直观了

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int len1 = text1.size();
        int len2 = text2.size();
        vector<vector<int>> dp(len1 + 1,vector<int>(len2 + 1));
        for(int i = 1;i <= len1;i++) {
            for(int j = 1;j <= len2;j++) {
                if(text1[i - 1] == text2[j - 1])
                    // 当前考虑的字符相等，则考虑两边去除相等的这个字符时的子串
                    // 目前状态下的最长公共字串长度 = 两边去除相等的这个字符时的子串的最长公共字串长度 + 1
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                else
                    // 当前考虑的字符不相等，则从两边的前一个字串继承最长公共字串长度，选其最大者
                    dp[i][j] = max(dp[i - 1][j],dp[i][j - 1]);
            }
        }
        return dp[len1][len2];
    }
};

int main() {
    // should be 3
    cout << Solution().longestCommonSubsequence("abcde","ace") << '\n';
    // should be 2
    cout << Solution().longestCommonSubsequence("abc","def") << '\n';
    return 0;
}
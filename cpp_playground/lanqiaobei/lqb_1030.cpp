// https://www.lanqiao.cn/problems/1030/learning/?page=1&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B,DFS&sort=pass_rate&asc=0
// 基本上就是leetcode_longest_common_subsequence换皮
// lqb给加了个dfs的tag，意义不明
// 100%

#include <bits/stdc++.h>

using namespace std;

int main() {
    ios::sync_with_stdio(false);

    // 分割
    array<vector<string>,2> a;
    string s;
    for(int i = 0;i < 2;i++) {
        cin >> s;
        int last = 0;
        int len = s.size();
        for(int j = 1;j < len;j++) {
            if(s[j] >= 'A' && s[j] <= 'Z') {
                a[i].emplace_back(s.substr(last,j - last));
                last = j;
            }
        }
        // emplace_back the last one
        a[i].emplace_back(s.substr(last,len - last));
    }

    // dp
    int len1 = a[0].size();
    int len2 = a[1].size();
    vector<vector<int>> dp(len1 + 1,vector<int>(len2 + 1));
    for(int i = 1;i <= len1;i++) {
        for(int j = 1;j <= len2;j++) {
            if(a[0][i - 1] == a[1][j - 1])
                dp[i][j] = dp[i - 1][j - 1] + 1;
            else
                dp[i][j] = max(dp[i - 1][j],dp[i][j - 1]);
        }
    }
    
    cout << dp[len1][len2] << '\n';
    return 0;
}
// https://leetcode.cn/problems/edit-distance/solutions/6455/zi-di-xiang-shang-he-zi-ding-xiang-xia-by-powcai-3/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    int minDistance(string word1, string word2) {
        vector<vector<int>> dp(word1.size() + 1,vector<int>(word2.size() + 1,0));
        for(int i = 0;i <= word2.size();i++)
            dp[0][i] = i;
        for(int i = 0;i <= word1.size();i++)
            dp[i][0] = i;
        for(int i = 1;i <= word1.size();i++) {
            for(int j = 1;j <= word2.size();j++) {
                if(word1[i - 1] == word2[j - 1])
                    dp[i][j] = dp[i - 1][j - 1];
                else
                    dp[i][j] = mino3(dp[i - 1][j - 1],dp[i - 1][j],dp[i][j - 1]) + 1;
            }
        }
        int res = dp.back().back();
        return res;
    }
private:
    int mino3(int a,int b,int c) {
        return min(min(a,b),c);
    }
};

int main() {
    cout << Solution().minDistance("horse","ros") << '\n';
    return 0;
}
// https://www.dotcpp.com/oj/problem3152.html

#include <bits/stdc++.h>

using namespace std;

/**
 * https://www.dotcpp.com/oj/submit_status.php?sid=15731705
 * 运行时间: 56ms    消耗内存: 2088KB
*/

int main() noexcept {
    int n;
    scanf("%d",&n);

    array<int,10> dp{0};

    int maxn = INT_MIN;
    for(int i = 1;i <= n;i++) {
        string cur;
        cin >> cur;

        int cur_head = cur[0] - '0';
        int cur_tail = cur.back() - '0';

        //                           discard current   or not
        // dp[i][cur_tail] = max( dp[i - 1][cur_tail], dp[i - 1][cur_head] + 1 )
        dp[cur_tail] = max(dp[cur_tail],dp[cur_head] + 1);
        maxn = max(maxn,dp[cur_tail]);
    }

    cout << n - maxn << endl;
    return 0;
}
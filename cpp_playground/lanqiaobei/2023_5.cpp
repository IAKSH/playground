// https://www.dotcpp.com/oj/problem3154.html

#include <bits/stdc++.h>

using namespace std;

/**
 * https://www.dotcpp.com/oj/submit_status.php?sid=15744503
 * 运行时间: 45ms    消耗内存: 3608KB
*/

int main() noexcept {
    int k;
    string s;
    char c1,c2;
    cin >> k >> s >> c1 >> c2;

    int len = s.size();
    int c1_cnt = 0;
    long long cnt = 0;

    for(int i = k - 1;i < len;i++) {
        c1_cnt += s[i - k + 1] == c1;
        if(s[i] == c2) {
            cnt += c1_cnt;
        }
    }

    cout << cnt << '\n';
    return 0;
}
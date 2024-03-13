// https://www.dotcpp.com/oj/problem3154.html

#include <bits/stdc++.h>

using namespace std;

/**
 * 超时
 * 
 * https://www.dotcpp.com/oj/submit_status.php?sid=15743322
 * 运行时间: 1561ms    消耗内存: 3608KB
*/

int main() noexcept {
    int k;
    string s;
    char c1,c2;
    cin >> k >> s >> c1 >> c2;

    int len = s.size();
    int cnt = 0;

    for(int i = 0;i < len;i++) {
        if(s[i] == c1) {
            for(int j = i;j < len;j++) {
                if(s[j] == c2 && j - i + 1 >= k) {
                    ++cnt;
                }
            }
        }
    }

    cout << cnt << '\n';
    return 0;
}
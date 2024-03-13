// https://www.dotcpp.com/oj/problem3154.html

#include <bits/stdc++.h>

using namespace std;

/**
 * 部分通过
 * 
 * https://www.dotcpp.com/oj/submit_status.php?sid=15743570
 * 运行时间: 348ms    消耗内存: 5180KB
*/

int main() noexcept {
    int k;
    string s;
    char c1,c2;
    cin >> k >> s >> c1 >> c2;

    int len = s.size();
    int cnt = 0;

    vector<int> heads;
    for(int i = 0;i < len;i++) {
        if(s[i] == c1) {
            heads.emplace_back(i);
        }
        else if(s[i] == c2) {
            for(const auto& j : heads) {
                cnt += i - j + 1 >= k;
            }
        }
    }

    cout << cnt << '\n';
    return 0;
}
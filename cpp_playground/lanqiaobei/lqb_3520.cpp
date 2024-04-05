// https://www.lanqiao.cn/problems/3520/learning/?page=3&first_category_id=1&tags=2023

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    int len,cnt,d;cin >> d;
    string t,s;
    for(int i = 0;i < d;i++) {
        cin >> t >> s;
        len = t.size();
        cnt = 0;
        for(int j = 0;j < len && cnt != -1;j++) {
            if(t[j] != s[j]) {
                if(j < 1 || j >= len || s[j - 1] == s[j] || s[j + 1] == s[j]) {
                    //cout << -1 << '\n';
                    cnt = -1;
                    break;
                }
                ++cnt;
            }
        }
        cout << cnt << '\n';
    }
    return 0;
}
// https://www.lanqiao.cn/problems/136/learning/?page=1&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B
// 能debug的话就做起来很无脑
// 100%

#include <bits/stdc++.h>

using namespace std;

string trans(string& s) {
    int len = s.size();
    string res = "";

    if(len == 1) {
        res += '1';
        res += s[0];
    }
    else {
        int cnt = 1;
        char last = s[0];
        for(int i = 1;i < len;i++) {
            if(s[i] == last)
                ++cnt;
            else {
                res += '0' + cnt;
                res += last;
                last = s[i];
                cnt = 1;
            }
        }
        // 处理最后一个
        res += '0' + cnt;
        res += last;
    }
    return res;
}

int main() {
    ios::sync_with_stdio(false);

    string s;
    int n;
    cin >> s >> n;

    while(n-- > 0)
        s = trans(s);
    
    cout << s << '\n';
    return 0;
}
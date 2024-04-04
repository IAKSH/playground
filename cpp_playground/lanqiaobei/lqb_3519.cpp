/**
 * https://www.lanqiao.cn/problems/3519/learning/?page=1&first_category_id=1&second_category_id=3&difficulty=20&tags=2023
 * 根本不需要DP
 * 直接线性推过去
 * 只需要将?视为通用即可，无论是一个?还是两个?，也无论这一个?在哪儿，只要有?就能组成一个新的字串。
*/

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    string s;cin >> s;
    int cnt = 0;
    int len = s.size();
    array<char,2> sub;
    for(int i = 0;i < len - 1;i++) {
        copy(s.begin() + i,s.begin() + i + 2,sub.begin()); 
        bool b = ((find(sub.begin(),sub.begin() + 2,'?') != sub.end()) || (sub[0] == sub[1]));
        cnt += b;
        i += b;
    }
    cout << cnt << '\n';
    return 0;
}

/*
int main() noexcept {
    string s;cin >> s;
    bool begin_type;
    int sub_cnt = 0;
    int cnt = 0;
    int len = s.size();
    for(int i = 0;i < len;i++) {
        if(sub_cnt == 0) {
            if(s[i] == '?') {
                // TODO: 需要判断是否到边界，这需要改最上面的for
                if(i != len - 1) {
                    begin_type = (s[i + 1] == '1');
                    ++sub_cnt;
                }
                else
                    sub_cnt = 0;// 也许不需要
            }
            else {
                begin_type = (s[i] == '1');
                ++sub_cnt;
            }
        }
        else {
            if(sub_cnt < 2) {
                if(s[i] == '?' || ((s[i] == '0') ^ begin_type))
                    ++sub_cnt;
            }
            else {
                sub_cnt = 0;
                ++cnt;
            }
        }
    }
    cout << cnt << '\n';
    return 0;
}
*/
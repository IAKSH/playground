// https://www.lanqiao.cn/problems/2142/learning/?page=2&first_category_id=1&second_category_id=3&tags=2022,%E7%9C%81%E8%B5%9B

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    array<int,26> cnts{0};
    string s;cin >> s;
    for(const auto& c : s)
        ++cnts[c - 'A'];
    //cout << static_cast<char>(max_element(cnts.begin(),cnts.end()) - cnts.begin() + 'A') << '\n';
    int maxn = *max_element(cnts.begin(),cnts.end());
    for(int i = 0;i < 26;i++)
        if(cnts[i] == maxn)
            cout << static_cast<char>(i + 'A');
    cout << '\n';
    return 0;
}
// https://www.lanqiao.cn/problems/170/learning/?page=2&first_category_id=1&second_category_id=3&difficulty=20

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    array<int,26> cnts{0};
    string s;cin >> s;
    for(const auto& c : s)
        ++cnts[c - 'a'];

    int maxn = INT_MIN;
    int minn = INT_MAX;
    for(const auto& i : cnts) {
        if(i > 0) {
            maxn = max(maxn,i);
            minn = min(minn,i);   
        }
    }

    cout << maxn - minn << '\n';
    return 0;
}
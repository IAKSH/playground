// https://www.lanqiao.cn/problems/2411/learning/?page=1&first_category_id=1&second_category_id=3&difficulty=20&tags=2023

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    int w,n;cin >> w >> n;
    int res = (w + n) % 7;
    cout << (res == 0 ? 7 : res) << '\n';
    return 0;
}
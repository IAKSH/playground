// https://www.lanqiao.cn/problems/143/learning/?page=1&first_category_id=1&second_category_id=3&difficulty=20

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    int n;cin >> n;
    int cnt = n;
    while(n >= 3) {
        ++cnt;
        n -= 2;
    }
    cout << cnt << '\n';
    return 0;
}
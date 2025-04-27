// https://www.lanqiao.cn/problems/152/learning/?page=1&first_category_id=1&second_category_id=3&difficulty=20

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    int n,a,b,c;
    cin >> n >> a >> b >> c;

    int cnt = 0;
    for(int i = 1;i <= n;i++) {
        cnt += (i % a != 0 && i % b != 0 && i % c != 0);
    }

    cout << cnt << '\n';
    return 0;
}
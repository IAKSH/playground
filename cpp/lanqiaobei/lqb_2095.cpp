// https://www.lanqiao.cn/problems/2095/learning/?page=1&first_category_id=1&second_category_id=3&tags=2022

#include <bits/stdc++.h>

using namespace std;

int base9ToDec(int n) noexcept {
    int dec = 0;
    int i = 0;
    while(n > 0) {
        dec += pow(9,i++) * (n % 10);
        n /= 10;
    }
    return dec;
}

int main() noexcept {
    //int n;cin >> n;
    //cout << base9ToDec(n) << '\n';
    cout << base9ToDec(2022) << '\n';
    return 0;
}
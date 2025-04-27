// https://www.lanqiao.cn/problems/606/learning/?page=1&first_category_id=1&second_category_id=3&name=%E6%95%B0%E7%9A%84%E5%88%86%E8%A7%A3

#include <bits/stdc++.h>

using namespace std;

bool legal(int n) noexcept {
    while(n > 0) {
        if(n % 10 == 2 || n % 10 == 4) {
            return false;
        }
        n /= 10;
    }
    return true;
}

int main() noexcept {
    int cnt = 0;
    for(int i = 1;i < 2019;i++) {
        for(int j = i + 1;j < 2019;j++) {
            int k = 2019 - i - j;
                if(j < k && legal(i) && legal(j) && legal(k)) {
                ++cnt;
                //cout << i << ' ' << j << ' ' << k << '\n';
            }
        }
    }
    cout << cnt << '\n';
    return 0;
}
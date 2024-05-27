// https://www.lanqiao.cn/problems/1028/learning/?page=1&first_category_id=1&sort=pass_rate&asc=0&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B
// 为什么这也能算困难?

#include <bits/stdc++.h>

using namespace std;

int main() {
    ios::sync_with_stdio(false);

    int i,j,cnt = 0;
    for(i = 1;i <= 2020;i++) {
        for(j = i - 1;j > 1;j--) {
            if(i % j == 0) {
                ++cnt;
                //cout << i << '\t' << j << '\n';
                break;
            }
        }
    }

    cout << cnt << '\n';
    return 0;
}
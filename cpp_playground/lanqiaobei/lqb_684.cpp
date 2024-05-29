// https://www.lanqiao.cn/problems/684/learning/?page=1&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B&sort=pass_rate&asc=0
// 无脑暴搜

#include <bits/stdc++.h>

using namespace std;

int main() {
    ios::sync_with_stdio(false);
    int i,j,k;
    for(i = 0;i < 100;i++) {
        for(j = 0;j < 100;j++) {
            for(k = 0;k < 100;k++) {
                if(3 * i + 7 * j + k == 315 && 4 * i + 10 * j + k == 420) {
                    //cout << i << '\t' << j << '\t' << k << '\t' << i + j + k << '\n';
                    // 实际上有多组结果，但是合都为105
                    cout << i + j + k << '\n';
                    return 0;
                }
            }
        }
    }
    //return 0;
    return 1;
}
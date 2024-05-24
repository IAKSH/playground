// https://www.lanqiao.cn/problems/690/learning/?page=1&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B&sort=pass_rate&asc=0
// 数据太小，直接暴力
// 需要注意的是a + b + c = 100，第一遍的时候没注意

#include <bits/stdc++.h>

using namespace std;

int main() {
    int a,b,c;
    for(a = 0;a < 100;a++) {
        for(b = 0;b < 100 - a;b++) {
            c = 100 - a - b;
            if(8 * a + 6 * b + 4 * c == 600 && a + 3 * b + 4 * c == 280) {
                //cout << a << '\n' << b << '\n' << c << '\n';
                cout << b << '\n';
            }
        }
    }

    return 0;
}
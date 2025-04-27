// https://www.lanqiao.cn/problems/3495/learning/?page=1&first_category_id=1&difficulty=20&tags=2023

#include <bits/stdc++.h>

using namespace std;

bool check(int y,int m,int d) noexcept {
    //int lcm = m * d / __gcd(m,d);
    //return y % lcm == 0;
    return (y % m == 0) && (y % d == 0);
}

bool isLeapYaer(int y) noexcept {
    return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
}

int main() noexcept {
    array<int,12> days_in_month{31,28,31,30,31,30,31,31,30,31,30,31};
    int cnt = 0;
    for(int y = 2000;y <= 2000000;y++) {
        for(int m = 1;m <= 12;m++) {
            int days = (m == 2 && isLeapYaer(y)) ? 29 : days_in_month[m - 1];
            for(int d = 1;d <= days;d++) {
                cnt += check(y,m,d);
                if(y == 2000000 && m == 1 && d == 1) {
                    cout << cnt << '\n';
                    return 0;
                }
            }
        }
    }
    return 1;
}
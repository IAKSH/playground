// https://www.lanqiao.cn/problems/2408/learning/?page=1&first_category_id=1&difficulty=20&tags=2023

#include <bits/stdc++.h>

using namespace std;

int acc(int n) noexcept {
    int sum = 0;
    while(n > 0) {
        sum += n % 10;
        n /= 10;
    }
    return sum;
}

bool isLeapYaer(int y) noexcept {
    return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
}

int main() noexcept {
    array<int,12> days_in_month{31,28,31,30,31,30,31,31,30,31,30,31};
    int cnt = 0;
    for(int y = 1900;y <= 9999;y++) {
        for(int m = 1;m <= 12;m++) {
            int days = (m == 2 && isLeapYaer(y)) ? 29 : days_in_month[m - 1];
            for(int d = 1;d <= days;d++)
                cnt += (acc(y) == (acc(m) + acc(d)));
        }
    }
    cout << cnt << '\n';
    return 0;
}
// https://www.lanqiao.cn/problems/2096/learning/?page=1&first_category_id=1&second_category_id=3&tags=2022

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    array<int,12> days{31,28,31,30,31,30,31,31,30,31,30,31};
    array<int,6> date{2,2,0,0,0,0};//yymmdd，因为只讨论连续三个数字，所以忽略前两位
    int cnt = 0;
    for(int i = 1;i <= 12;i++) {
        for(int j = 1;j < days[i - 1];j++) {
            date[2] = i / 10;
            date[3] = i % 10;
            date[4] = j / 10;
            date[5] = j % 10;
            for(int k = 0;k < 4;k++)
                cnt += (date[k] + 1 == date[k + 1] && date[k] + 2 == date[k + 2]);
        }
    }
    cout << cnt << '\n';
    return 0;
}
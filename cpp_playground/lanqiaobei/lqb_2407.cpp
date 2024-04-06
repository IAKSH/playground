// https://www.lanqiao.cn/problems/2407/learning/?isWithAnswer=true&page=1&first_category_id=1&tags=2023

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    //cout << 2022 / 26 << '\n';
    cout << static_cast<char>('A' + (2022 / 26) / 26 - 1);
    cout << static_cast<char>('A' + (2022 / 26) % 26 - 1);
    cout << static_cast<char>('A' + 2022 % 26 - 1) << '\n';
    return 0;
}
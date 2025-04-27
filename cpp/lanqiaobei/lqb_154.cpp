// https://www.lanqiao.cn/problems/154/learning/?page=1&first_category_id=1&second_category_id=3&difficulty=20

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    string s;cin >> s;
    for(const auto& c : s)
        cout << static_cast<char>(c + 3 > 'z' ? c - 23 : c + 3);
    cout << '\n';
    return 0;
}
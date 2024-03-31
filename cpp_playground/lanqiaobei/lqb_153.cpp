// https://www.lanqiao.cn/problems/153/learning/?page=1&first_category_id=1&second_category_id=3&difficulty=20

#include <bits/stdc++.h>

using namespace std;

bool legel(int n) noexcept {
    while(n > 0) {
        if(n % 10 == 2)
            return false;
        n /= 10;
    }
    return true;
}

bool legel_str(int n) noexcept {
    for(const auto& c : to_string(n))
        if(c == '2')
            return false;
    return true;
}

int main() noexcept {
    int n,cnt = 0;cin >> n;
    for(int i = 1;i <= n;i++)
        cnt += legel(i);
    cout << cnt << '\n';
    return 0;
}
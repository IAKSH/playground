// https://www.lanqiao.cn/problems/6269/learning/?page=4&first_category_id=1&tags=2023

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    const char* answers[] {"low","mid","high"};
    int p;cin >> p;
    cout << answers[p % 3] << '\n';
    return 0;
}
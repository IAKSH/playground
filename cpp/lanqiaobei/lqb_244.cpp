// https://www.lanqiao.cn/problems/244/learning/?page=4&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B&sort=pass_rate&asc=0
// 怎么这么简单？

#include <bits/stdc++.h>

using namespace std;

int main() {
    ios::sync_with_stdio(false);
    
    string s,t;
    cin >> s >> t;

    int cnt = 0;
    auto t_it = t.begin();
    for(const auto& c : s) {
        if(t_it == t.end())
            break;
        if(c == *t_it) {
            ++cnt;
            ++t_it;
        }
    }

    cout << cnt << '\n';
    return 0;
}
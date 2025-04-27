/**
 * https://www.lanqiao.cn/problems/163/learning/?page=1&first_category_id=1&second_category_id=3&difficulty=20
 * 一例超时
*/

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    int m,n,h;
    cin >> n >> m;
    vector<vector<int>> mat(n,vector<int>(m));
    for(auto& line : mat)
        for(auto& i : line)
            cin >> i;
    cin >> h;

    int cnt = 0;
    for(int i = 0;i < h;i++) {
        for(const auto& line : mat)
            for(const auto& item : line)
                cnt += (item >= i + 1);
        cout << cnt << '\n';
    }
    return 0;
}
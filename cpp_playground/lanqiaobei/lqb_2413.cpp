// https://www.lanqiao.cn/problems/2413/learning/?page=1&first_category_id=1&second_category_id=3&tags=2023

#include <bits/stdc++.h>

using namespace std;

struct Clean {
    int r1,r2,c1,c2;
};

int main() noexcept {
    int n,m,t;cin >> n >> m >> t;
    vector<Clean> v(t);
    for(auto& c : v)
        cin >> c.r1 >> c.c1 >> c.r2 >> c.c2;

    vector<vector<bool>> mat(n,vector<bool>(m,true));
    for(const auto& c : v) {
        for(int i = c.r1;i <= c.r2;i++) {
            for(int j = c.c1;j <= c.c2;j++)
                mat[i - 1][j - 1] = false;
        }
    }

    int cnt = 0;
    for(const auto& line : mat) {
        for(const auto& b : line)
            cnt += b;
    }

    cout << cnt << '\n';
    return 0;
}
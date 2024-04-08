// https://www.lanqiao.cn/problems/549/learning/?page=1&first_category_id=1&second_category_id=3&tags=2022

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    int n,m;cin >> n >> m;
    vector<vector<int>> mat(n,vector<int>(m));
    for(auto& line : mat) {
        for(auto& i : line)
            cin >> i;
    }

    cout << '\n';

    int cnt;
    for(int i = 0;i < n;i++) {
        for(int j = 0;j < m;j++) {
            if(mat[i][j]) {
                cout << 9 << ' ';
                continue;
            }
            cnt = 0;
            // TODO: 简化if
            // 为什么只有一个测试例，直接过了
            // 懒了，不改了
            if(i > 0)
                cnt += mat[i - 1][j];
            if(i < n - 1)
                cnt += mat[i + 1][j];
            if(j > 0)
                cnt += mat[i][j - 1];
            if(j < m - 1)
                cnt += mat[i][j + 1];
            if(i < n - 1 && j > 0)
                cnt += mat[i + 1][j - 1];
            if(i < n - 1 && j < m - 1)
                cnt += mat[i + 1][j + 1];
            if(i > 0 && j > 0)
                cnt += mat[i - 1][j - 1];
            if(i > 0 && j < m - 1)
                cnt += mat[i - 1][j + 1];
            cout << cnt << ' ';
        }
        cout << '\n';
    }
    return 0;
}
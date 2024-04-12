// https://www.lanqiao.cn/problems/196/learning/?page=1&first_category_id=1&second_category_id=3&tags=2019

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    int t;
    cin >> t;
    vector<vector<int>> v(t);
    for(int i = 0;i < t;i++) {
        int n;
        cin >> n;
        v[i].resize(n);
        for(int j = 0;j < n;j++) {
            cin >> v[i][j];
        }
    }
}
// https://www.lanqiao.cn/problems/155/learning/?page=1&first_category_id=1&second_category_id=3&difficulty=20

#include <bits/stdc++.h>

using namespace std;

int distance(const vector<int>& v,int i,int j) noexcept {
    return abs(i - j) + abs(v[i] - v[j]);
}

int main() noexcept {
    int n,maxn = INT_MIN;cin >> n;
    vector<int> v(n);
    for(auto& i : v)
        cin >> i;

    for(int i = 0;i < n;i++) {
        for(int j = 0;j < n;j++) {
            maxn = max(maxn,distance(v,i,j));
        }
    }

    cout << maxn << '\n';
    return 0;
}
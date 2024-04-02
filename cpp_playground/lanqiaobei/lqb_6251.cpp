// https://www.lanqiao.cn/problems/6251/learning/?first_category_id=1&page=1&second_category_id=3&difficulty=20&tags=2023
// TODO: 一个没过🤣

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    int n,k;cin >> n >> k;
    vector<int> v(n);
    for(auto& i : v) cin >> i;

    float sum = 0;
    int cnt = 0;// 也许有办法直接计算出cnt
    for(int i = 0;i <= n - k;i++) {
        auto minmax = minmax_element(v.begin() + i,v.begin() + i + k);
        sum += *minmax.second - *minmax.first;
        ++cnt;
    }
    printf("%.2f\n",sum / cnt);
    return 0;
}
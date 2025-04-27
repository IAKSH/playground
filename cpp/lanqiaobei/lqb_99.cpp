/**
 * https://www.lanqiao.cn/problems/99/learning/?page=1&first_category_id=1&second_category_id=3
 * 通过率：62.5%
*/

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    int n,k,maxn = 0;
    vector<pair<int,int>> chocolates;

    cin >> n >> k;
    for(int i = 0;i < n;i++) {
        int h,w;
        scanf("%d%d",&h,&w);
        maxn = max(maxn,max(h,w));
        chocolates.emplace_back(pair<int,int>(h,w));
    }

    int lsq;
    for(lsq = maxn;lsq > 0;lsq--) {
        int _k = 0;
        for(const auto& cho : chocolates)
            _k += cho.first / lsq * cho.second / lsq;
        if(_k >= k)
            break;
    }

    cout << lsq << '\n';
    return 0;
}
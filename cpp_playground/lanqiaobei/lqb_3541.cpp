// https://www.lanqiao.cn/problems/3541/learning/?page=3&first_category_id=1&tags=2023
// 0% abandoned

#include <bits/stdc++.h>

using namespace std;

int price(int k,int x,int b) noexcept {
    return x == 0 ? 0 : max(k * x + b,0);
}

int main() noexcept {
    int n,m;cin >> n >> m;
    vector<pair<int,int>> v(m);
    for(auto& p : v)
        cin >> p.first >> p.second;
    
    sort(v.begin(),v.end(),[](const pair<int,int>& p0,const pair<int,int>& p1){
        return p0.second - p0.first > p1.second - p1.first;
    });

    int max_price = INT_MIN;
    vector<int> man_cnt(m,0);
    
    for(int i = 0;i < n;i++) {
        fill(man_cnt.begin() + 1,man_cnt.end(),0);
        man_cnt[0] = i;
        for(int j = 0;j < m - 1;j++) {
            while(man_cnt[j] >= 0) {
                int total_price = 0;
                for(const auto& man : man_cnt)
                    total_price += price(v[j].first,man,v[j].second);
                max_price = max(max_price,total_price);

                // debug
                //for(const auto& man : man_cnt) {
                //    cout << man << ',';
                //}
                //cout << '\b' << " = " <<  total_price << '\n';

                if(man_cnt[j] > 0) {
                    --man_cnt[j];
                    ++man_cnt[j + 1]; 
                }
                else
                    break;
            }
        }
    }

    cout << max_price << '\n';
    return 0;
}

/*
4 2
-4 10
-2 7
=12
*/

//for(int i = 0;i < m;i++) {
//    int maxn = INT_MIN;
//    for(int j = 1;j < n;j++) {
//        int h = max(v[i].first * j + v[i].second,0);
//        maxn = max(maxn,h);
//    }
//    sum += maxn;
//}
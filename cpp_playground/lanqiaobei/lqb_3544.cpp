// https://www.lanqiao.cn/problems/3544/learning/?page=1&first_category_id=1&second_category_id=3&tags=2023,%E4%BA%8C%E5%88%86&difficulty=30
// 未经测试

#include <bits/stdc++.h>

using namespace std;

// 模拟，等下再写个区间合并+二分查找

int main() noexcept {
    int n,len;cin >> n >> len;
    vector<bool> pipe(len,false);
    vector<pair<int,int>> v(n);
    for(auto& p : v)
        cin >> p.first >> p.second;
    
    int t = 0;
    for(;true;t++) {
        bool flag = true;
        for(const auto& b : pipe)
            if(!b) flag = false;
        if(flag) {
            cout << t << '\n';
            return 0;
        }
        for(const auto& p : v) {
            if(p.second <= t) {
                for(int i = p.first - (t - p.second);i <= p.first + (t - p.second);i++) {
                    if(i < 1 && i > n) continue;
                    pipe[i - 1] = true;
                }
            }
        }
    }
    return 1;
}
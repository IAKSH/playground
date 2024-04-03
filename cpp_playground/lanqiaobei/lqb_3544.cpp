// https://www.lanqiao.cn/problems/3544/learning/?page=1&first_category_id=1&second_category_id=3&tags=2023,%E4%BA%8C%E5%88%86&difficulty=30
// 未经测试
// 模拟，等下再写个区间合并+二分查找
// 不出意外的超时了，5过5超时

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    int n,len;cin >> n >> len;
    vector<bool> pipe(len,false);
    vector<pair<int,int>> v(n);
    for(auto& p : v)
        cin >> p.first >> p.second;
    
    int cur_t = 1;
    while(true) {
        bool flag = true;
        for(const auto& b : pipe)
            if(!b) flag = false;
        if(flag) {
            cout << cur_t - 1 << '\n';
            break;
        }
        for(const auto& p : v) {
            if(p.second <= cur_t) {
                for(int i = p.first - (cur_t - p.second);i <= p.first + (cur_t - p.second);i++) {
                    if(i > 0 && i <= len)
                        pipe[i - 1] = true;
                }
            }
        }
        ++cur_t;
    }
    return 0;
}
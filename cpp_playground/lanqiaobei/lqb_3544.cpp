// https://www.lanqiao.cn/problems/3544/learning/?page=1&first_category_id=1&second_category_id=3&tags=2023,%E4%BA%8C%E5%88%86&difficulty=30
// 90% 1超时

#include <bits/stdc++.h>

using namespace std;

int n,len;
vector<bool> pipe;
vector<pair<int,int>> v;
vector<bool> check_v;

bool check(int t) noexcept {
    fill(check_v.begin(),check_v.end(),false);
    for(const auto& p : v) {
        int range = t - p.second;
        if(range >= 0) {
            // 左闭右开
            int cur_l = max( p.first - range - 1,0);
            int cur_r = min(p.first + range,len);
            fill(check_v.begin() + cur_l,check_v.begin() + cur_r,true);
        }
    }
    for(const auto& b : check_v) {
        if(!b) return false;
    }
    return true;
}

int main() noexcept {
    cin >> n >> len;
    pipe.resize(len,false);
    v.resize(n);
    check_v.resize(len);
    for(auto& p : v)
        cin >> p.first >> p.second;

    int l = 0;
    int r = len + max_element(v.begin(),v.end(),[](const pair<int,int>& p1,const pair<int,int>& p2){return p1.second > p2.second;})->second;
    while(l != r) {
        int mid = (l + r) / 2;
        if(check(mid)) {
            r = mid;
        }
        else {
            l = mid + 1;
        }
    }
    cout << l << '\n';
    return 0;
}

/*
3 10
1 1
6 5
10 2
= 5

1 100
1 99
=198
*/
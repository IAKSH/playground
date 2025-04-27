/**
 * https://www.lanqiao.cn/problems/3505/learning/?page=1&first_category_id=1&second_category_id=3&difficulty=10&tags=2023
 * 纯dfs会超时，尝试从dfs改造bfs
 * 为什么两种都只能过55%，bfs甚至有段错误
 * 懒得改了
*/

#include <bits/stdc++.h>

using namespace std;

static int n,m;
static vector<float> v;

int min_cut_cnt = INT_MAX;

void dfs(float sum,int dn,int cut_cnt) noexcept {
    if(dn <= n && cut_cnt <= min_cut_cnt) {
        if(sum == m)
            min_cut_cnt = min(min_cut_cnt,cut_cnt);
        else if(sum < m) {
            dfs(sum,dn + 1,cut_cnt);
            dfs(sum + v[dn] / 2.0f,dn + 1,cut_cnt + 1);
            dfs(sum + v[dn],dn + 1,cut_cnt);
        }
    }
}

int main() noexcept {
    cin >> n >> m;
    v.resize(n);
    for(auto& i : v)
        cin >> i;

    sort(v.begin(),v.end(),greater<float>());

    dfs(0,0,0);
    dfs(v[0] / 2.0f,0,1);
    dfs(v[0],0,0);

    cout << min_cut_cnt << '\n';
    return 0;
}
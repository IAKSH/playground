// https://www.lanqiao.cn/problems/1019/learning/?page=5&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B&sort=pass_rate&asc=0
// 本来应该dfs，结果最后直接遍历然后计算xy距离合过了

#include <bits/stdc++.h>

using namespace std;

#ifdef USE_DFS

vector<vector<int>> m(4021,vector<int>(4021));

// 跑一年
void dfs(int last_dx,int last_dy,int x,int y,int depth) {
    cout << x << '\t' << y << '\t' << depth << '\n';
    if(depth == 2020)
        return;
    m[y][x] = 1;
    if(x > 0 && last_dx != 1)
        dfs(-1,0,x - 1,y,depth + 1);
    if(x < 4020 && last_dx != -1)
        dfs(1,0,x + 1,y,depth + 1);
    if(y > 0 && last_dy != 1)
        dfs(0,-1,x,y - 1,depth + 1);
    if(y < 4020 && last_dy != -1)
        dfs(0,1,x,y + 1,depth + 1);
}

int main() {
    ios::sync_with_stdio(false);

    dfs(0,0,0,0,0);
    dfs(0,0,2020,11,0);
    dfs(0,0,11,14,0);
    dfs(0,0,2000,2000,0);

    int cnt = 0;
    for(const auto& v : m)
        //accumulate(v.begin(),v.end(),cnt);
        for(const auto& i : v)
            cnt += i;

    cout << cnt << '\n';
    return 0;
}

#else 

int distance(int x1,int y1,int x2,int y2) {
    //return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    return abs(x1 - x2) + abs(y1 - y2);
}

int main() {
    ios::sync_with_stdio(false);

    int cnt = 0;
    for(int i = 0;i < 10000;i++) {
        for(int j = 0;j < 10000;j++) {
            if(distance(i,j,5000,5000) <= 2020      // (0,0)
                || distance(i,j,2020 + 5000,11 + 5000) <= 2020
                || distance(i,j,11 + 5000,14 + 5000) <= 2020
                || distance(i,j,2000 + 5000,2000 + 5000) <= 2020
                )
                ++cnt;
        }
    }

    cout << cnt << '\n';
    return 0;
}

#endif
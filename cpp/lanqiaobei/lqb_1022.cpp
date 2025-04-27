//https://www.lanqiao.cn/problems/1022/learning/?page=1&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B,DFS&sort=pass_rate&asc=0
// 原本想在dfs中记录方向来做减枝
// 然后在题解里看见有直接染色的方式，想了一下好像确实更简单(尤其是代码实现上，会短很多)，性能也应该会好一点

#include <bits/stdc++.h>

using namespace std;

array<array<int,4>,4> m;
int cnt = 0;
int cur = 1;

void dfs(int x,int y) {
    if(cur == 16)
        ++cnt;
    else {
        m[y][x] = 1;
        ++cur;
        if(x > 0 && !m[y][x - 1])
            dfs(x - 1,y);
        if(x < 3 && !m[y][x + 1])
            dfs(x + 1,y);
        if(y > 0 && !m[y - 1][x])
            dfs(x,y - 1);
        if(y < 3 && !m[y + 1][x])
            dfs(x,y + 1);
        m[y][x] = 0;
        --cur;
    }
}

int main() {
    ios::sync_with_stdio(false);

    for(int i = 0;i < 4;i++) {
        for(int j = 0;j < 4;j++)
            dfs(j,i);
    }

    cout << cnt << '\n';
    return 0;
}
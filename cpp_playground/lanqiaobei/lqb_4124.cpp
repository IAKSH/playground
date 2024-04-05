/**
 * https://www.lanqiao.cn/problems/4124/learning/?page=4&first_category_id=1&tags=2023
 * 简单dfs，目测难点在9+16这两种糖果在2~5个/人上的枚举
 * ...大概是难得写(?
 * 似乎还有终止条件，因为是混合的
 * 感觉完全就是简单题的水平吧，只不过因为是填空题所以不太能捡分
 * 5067671
*/

#include <bits/stdc++.h>

using namespace std;

static int cnt = 0;

void dfs(int leftover_c1,int leftover_c2,int depth) noexcept {
    if(depth > 7 || leftover_c1 < 0 || leftover_c2 < 0)
        return;
    if(depth == 7 && leftover_c1 + leftover_c2 == 0) {
        ++cnt;
        return;
    }
    
    // got 2 candy
    dfs(leftover_c1 - 2,leftover_c2,depth + 1);
    dfs(leftover_c1,leftover_c2 - 2,depth + 1);
    dfs(leftover_c1 - 1,leftover_c2 - 1,depth + 1);
    // got 3 candy
    dfs(leftover_c1 - 3,leftover_c2,depth + 1);
    dfs(leftover_c1,leftover_c2 - 3,depth + 1);
    dfs(leftover_c1 - 2,leftover_c2 - 1,depth + 1);
    dfs(leftover_c1 - 1,leftover_c2 - 2,depth + 1);
    // got 4 candy
    dfs(leftover_c1 - 4,leftover_c2,depth + 1);
    dfs(leftover_c1,leftover_c2 - 4,depth + 1);
    dfs(leftover_c1 - 3,leftover_c2 - 1,depth + 1);
    dfs(leftover_c1 - 1,leftover_c2 - 3,depth + 1);
    dfs(leftover_c1 - 2,leftover_c2 - 2,depth + 1);
    // got 5 candy
    dfs(leftover_c1 - 5,leftover_c2,depth + 1);
    dfs(leftover_c1,leftover_c2 - 5,depth + 1);
    dfs(leftover_c1 - 4,leftover_c2 - 1,depth + 1);
    dfs(leftover_c1 - 1,leftover_c2 - 4,depth + 1);
    dfs(leftover_c1 - 3,leftover_c2 - 2,depth + 1);
    dfs(leftover_c1 - 2,leftover_c2 - 3,depth + 1);
}

int main() noexcept {
    int c1 = 9;
    int c2 = 16;
    dfs(c1,c2,0);
    cout << cnt << '\n';
    return 0;
}
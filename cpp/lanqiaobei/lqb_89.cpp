// https://www.lanqiao.cn/problems/89/learning/?page=1&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B
// 很平常的四联通dfs
// 实现得不太优雅
// 100%

#include <bits/stdc++.h>

using namespace std;

int n;
// 0->x 1->y
// real size = n
array<array<int,20>,2> arrows;
stack<int> w;

bool check_empty() {
    for(int i = 0;i < n;i++) {
        if(arrows[0][i] || arrows[1][i])
            return false;
    }
    return true;
}

// from: 0,1,2,3 -> up,down,left,right
void dfs(int x,int y,short from) {
    if(!arrows[0][x] || !arrows[1][y])
        return;
    --arrows[0][x];
    --arrows[1][y];
    w.emplace(x + y * n);
    
    if(x == n - 1 && y == n - 1 && check_empty()) {
        stack<int> res;
        while(!w.empty()) {
            res.emplace(w.top());
            w.pop();
        }
        while(!res.empty()) {
            cout << res.top() << ' ';
            res.pop();
        }
        cout << '\n';
        exit(0);
    }
    else {
        if(from != 0 && y - 1 >= 0)
            dfs(x,y - 1,1);
        if(from != 1 && y + 1 < n)
            dfs(x,y + 1,0);
        if(from != 2 && x - 1 >= 0)
            dfs(x - 1,y,3);
        if(from != 3 && x + 1 < n)
            dfs(x + 1,y,2);
    }

    ++arrows[0][x];
    ++arrows[1][y];
    w.pop();
}

int main() {
    ios::sync_with_stdio(false);

    cin >> n;

    for(int i = 0;i < 2;i++) {
        for(int j = 0;j < n;j++)
            cin >> arrows[i][j];
    }

    dfs(0,0,0);

    return 1;
}
// https://www.lanqiao.cn/problems/2193/learning/?page=8&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B&sort=pass_rate&asc=0
// 贪心的dfs，甚至有一点dp的感觉

#include <bits/stdc++.h>

using namespace std;

string n;
int a,b;
long long res = 0;

void dfs(int x,long long cur){
    int val = n[x] - '0';
    if(n[x]) { 
        int c = min(a,9 - val);
        a -= c;
        dfs(x + 1,cur * 10 + val + c);
        a += c;
        if(b > val){
            b = b - val - 1;
            dfs(x + 1,cur * 10 + 9);
            b = b + val + 1;
        }
    }else{
        res = max(res,cur);
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin >> n >> a >> b;

    dfs(0,0);

    cout << res << '\n';
    return 0;
}
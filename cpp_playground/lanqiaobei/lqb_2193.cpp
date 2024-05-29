// https://www.lanqiao.cn/problems/2193/learning/?page=8&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B&sort=pass_rate&asc=0
// nmd，还是得用字符串

#include <bits/stdc++.h>

using namespace std;

int n,a,b,m = 0,max_res = 0;

int opA(int x,int ind) {
    int val = x;
    for(int i = 0;i < ind;i++)
        val /= 10;
    val %= 10;
    int new_val = (val == 9 ? 0 : val + 1);
    return x - (val - new_val) * pow(10,ind);
}

int opB(int x,int ind) {
    int val = x;
    for(int i = 0;i < ind;i++)
        val /= 10;
    val %= 10;
    int new_val = (val == 0 ? 9 : val - 1);
    if(ind == m - 1 && new_val == 0)
        x += pow(10,m);
    return x - (val - new_val) * pow(10,ind);
}

void dfs(int x,int ind,bool op,int cur_a,int cur_b) {
    if(op)
        max_res = max(max_res,x = opA(x,ind));
    else
        max_res = max(max_res,x = opB(x,ind));
    if(ind + 1 < m) {
        if(cur_a)
            dfs(x,ind + 1,true,cur_a - 1,cur_b);
        if(cur_b)
            dfs(x,ind + 1,false,cur_a,cur_b - 1);
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin >> n >> a >> b;

    int tmp = n;
    while(tmp > 0) {
        tmp /= 10;
        ++m;
    }

    dfs(n,0,true,a,b);
    dfs(n,0,false,a,b);

    cout << max_res << '\n';
    return 0;
}
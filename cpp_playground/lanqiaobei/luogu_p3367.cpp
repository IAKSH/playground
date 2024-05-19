// https://www.luogu.com.cn/problem/P3367
// 2 AC 3 LTE 5 WA

#include <bits/stdc++.h>

using namespace std;

int main() {
    int n,m,z,x,y,i,j,k;
    char res[m],cnt = 0;
    cin >> n >> m;

    unordered_map<int,int> map;//val,group
    
    for(i = 0;i < n;i++)
        map[i + 1] = i + 1;

    for(i = 0;i < m;i++) {
        cin >> z >> x >> y;
        if(z == 1) {
            // merge
            auto& p1_i = map[x];
            auto& p2_i = map[y];
            for(auto& p : map) {
                if(p.second == p2_i)
                    p.second = p1_i;
            }
        }
        else {
            // check
            res[cnt++] = (map[x] == map[y] ? 'Y' : 'N');
        }
    }

    for(i = 0;i < cnt;i++)
        cout << res[i] << '\n';
    return 0;
}
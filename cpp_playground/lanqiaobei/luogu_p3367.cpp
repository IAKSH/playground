// https://www.luogu.com.cn/problem/P3367
// 8 RE 2 LTE
// 惨不忍睹

#include <bits/stdc++.h>

using namespace std;

int main() {
    int n,m,z,x,y,i,j,k;
    char res[m],cnt = 0;
    cin >> n >> m;

    vector<unordered_set<int>> v(n);
    for(i = 0;i < n;i++)
        v[i].emplace(i + 1);

    for(i = 0;i < m;i++) {
        cin >> z >> x >> y;
        for(j = 0;j < n && v[j].count(x) == 0;j++);
        for(k = 0;k < n && v[k].count(y) == 0;k++);
        if(z == 1) {
            // merge
            for(const auto& val : v[j])
                if(v[k].count(val) == 0)
                    v[k].emplace(val);
            v[j].clear();
        }
        else {
            // check
            res[cnt++] = (j == k ? 'Y' : 'N');
        }
    }

    for(i = 0;i < cnt;i++)
        cout << res[i] << '\n';
    return 0;
}
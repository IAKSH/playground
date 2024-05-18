// https://www.luogu.com.cn/problem/P1901

#include <bits/stdc++.h>

using namespace std;

// 不对，不能这样做
// 只能拿单调栈或者类似的东西来，不能下面这样只考虑靠近的两个
// 这道题必须要考虑遮挡问题。
// 下次继续了

int main() {
    cin.sync_with_stdio(false);

    int i,j,n,max_v;
    int h[3] = {0};
    int v[3] = {0};
    cin >> n;

    for(i = 0;i < n;i++) {
        if(i > 2 && i != n - 1) {
            for(j = 0;j < 2;j++) {
                h[j] = h[j + 1];
                v[j] = v[j + 1];
            }
        }
        cin >> h[(i > 2 ? 2 : i)] >> v[(i > 2 ? 2 : i)];
        if(i == 2) {
            max_v = v[0] + (h[0] > h[1] ? v[1] : 0);
        }
        else if(i == n - 1) {
            max_v = max(max_v,v[2] + (h[2] > h[1] ? v[1] : 0));
            
        }
        else {
            max_v = max(max_v,v[1] + (h[1] > h[0] ? v[0] : 0) + (h[1] > h[2] ? v[2] : 0));
        }
    }

    cout << max_v << '\n';
    return 0;
}

/*
3
4 2
3 5
6 10
*/
// https://www.luogu.com.cn/problem/P1901
// 单调栈
// AC

#include <bits/stdc++.h>

using namespace std;

int main() {
    ios::sync_with_stdio(false);

    int n;cin >> n;
    int h[n],v[n],res[n] = {0};
    stack<int> s;

    for(int i = 0;i < n;i++) {
        cin >> h[i] >> v[i];
        while(!s.empty() && h[s.top()] < h[i]) {
            // 从s.top()传到i
            res[i] += v[s.top()];
            s.pop();
        }
        if(!s.empty())
            // 从i传到s.top()
            res[s.top()] += v[i];
        s.emplace(i);
    }

    cout << *max_element(res,res + n) << '\n';
    return 0;
}

/*
3
4 2
3 5
6 10
*/
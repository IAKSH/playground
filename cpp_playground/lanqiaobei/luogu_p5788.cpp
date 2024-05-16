// https://www.luogu.com.cn/problem/P5788
// 6AC 4LTE

#include <bits/stdc++.h>

using namespace std;

// 先写一个无脑的O(n^2)

int main() {
    std::ios::sync_with_stdio(false);

    int n;cin >> n;
    std::vector<int> v(n);
    for(auto& i : v)
        cin >> i;

    int i,j;
    int len = v.size();
    bool flag;
    for(i = 0;i < len;i++) {
        flag = true;
        for(j = i + 1;j < len;j++) {
            if(v[i] < v[j]) {
                cout << j + 1 << ' ';
                flag = false;
                break;
            }
        }
        if(flag)
            cout << "0 ";
    }

    return 0;
}
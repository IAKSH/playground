// https://www.luogu.com.cn/problem/P3370
// 字符串哈希
// 因为STL，完全无脑了
// AC

#include <bits/stdc++.h>

using namespace std;

int main() {
    ios::sync_with_stdio(false);

    int n;cin >> n;
    unordered_set<string> set;
    string s;
    while(n-- > 0) {
        cin >> s;
        if(set.find(s) == set.end())
            set.emplace(s);
    }

    cout << set.size() << '\n';
    return 0;
}
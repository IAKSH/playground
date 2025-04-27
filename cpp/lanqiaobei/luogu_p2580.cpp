// https://www.luogu.com.cn/problem/P2580
// 用哈希表逃课
// 实际上应该是用字典树
// AC

#include <bits/stdc++.h>

using namespace std;

int main() {
    ios::sync_with_stdio(false);

    unordered_set<string> set,set1;
    int n; 
    cin >> n;

    string s;
    for(;n > 0;n--) {
        cin >> s;
        set.emplace(s);
    }

    cin >> n;
    for(;n > 0;n--) {
        cin >> s;
        if(set.find(s) != set.end()) {
            if(set1.empty() || set1.find(s) == set1.end()) {
                cout << "OK\n";
                set1.emplace(s);
            }
            else
                cout << "REPEAT\n";
        }
        else {
            cout << "WRONG\n";
        }
    }

    return 0;
}
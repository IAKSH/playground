// https://www.luogu.com.cn/problem/P8306
// 怎么还是炸了？？
// 暂存

#include <bits/stdc++.h>

using namespace std;

// 字典树在查找上只比哈希表节约一些内存吧，除此之外啥也不是
// 但是字典树能做前缀匹配，哈希表不能

// 复制的fusu的题解
// https://www.luogu.com.cn/article/uaqlf4xc
struct Node {
    int cnt;
    std::unordered_map<char, Node*> ch;
    
    Node() : cnt(0) {};
    
    void dfs() {
        for (auto [x, y] : ch) {
            y->dfs();
            cnt += y->cnt;
        }
    }
};

Node *rot;

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);
    int T, n, q;
    std::string s;
    for (std::cin >> T; T; --T) {
        std::cin >> n >> q;
        rot = new Node();
        for (int i = 1; i <= n; ++i) {
            std::cin >> s;
            auto u = rot;
            for (auto c : s) u = (u->ch[c] ? u->ch[c] : u->ch[c] = new Node);
            ++u->cnt;
        }
        rot->dfs();
        for (int i = 1; i <= q; ++i) {
            std::cin >> s;
            bool flag = true;
            auto u = rot;
            for (auto c : s) if (u->ch[c]) {
                u = u->ch[c];
            } else {
                flag = false; break;
            }
            std::cout << ((flag) ? u->cnt : 0) << '\n';
        }
    }
    return 0;
}
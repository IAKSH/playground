// https://www.luogu.com.cn/problem/P8306
// 怎么还是炸了？
// 暂存

#include <bits/stdc++.h>

using namespace std;

// 字典树在查找上只比哈希表节约一些内存吧，除此之外啥也不是
// 但是字典树能做前缀匹配，哈希表不能

// 支持A-Z a-z 0-9的基于数组的字典树
struct Trie {
private:
    struct TrieNode {
    int children[62];
    bool is_end;
        TrieNode() {
            memset(children, -1, sizeof(children));
        }
    };

public:
    vector<TrieNode> nodes;

    Trie() {
        nodes.push_back(TrieNode());
    }

    void insert(const string& str) {
        int nodeIndex = 0;
        for (char c : str) {
            int index = getIndex(c);
            if (nodes[nodeIndex].children[index] == -1) {
                nodes[nodeIndex].children[index] = nodes.size();
                nodes.push_back(TrieNode());
            }
            nodeIndex = nodes[nodeIndex].children[index];
        }
        nodes[nodeIndex].is_end = true;
    }

    bool search(const string& str) {
        int nodeIndex = 0;
        for (char c : str) {
            int index = getIndex(c);
            if (nodes[nodeIndex].children[index] == -1) {
                return false;
            }
            nodeIndex = nodes[nodeIndex].children[index];
        }
        return nodes[nodeIndex].is_end;
    }

    int prefixCount(const string& str) {
        int nodeIndex = 0;
        for (char c : str) {
            int index = getIndex(c);
            if (nodes[nodeIndex].children[index] == -1) {
                return 0;
            }
            nodeIndex = nodes[nodeIndex].children[index];
        }
        return countWords(nodeIndex);
    }

    vector<string> prefixMatch(const string& str) {
        vector<string> matches;
        int nodeIndex = 0;
        string prefix = "";
        for (char c : str) {
            int index = getIndex(c);
            if (nodes[nodeIndex].children[index] == -1) {
                return matches;
            }
            nodeIndex = nodes[nodeIndex].children[index];
            prefix += c;
            collectWords(nodeIndex, prefix, matches);
        }
        return matches;
    }

private:
    int getIndex(char c) {
        if(c >= 'A' && c <= 'Z') {
            return c - 'A';
        } else if(c >= 'a' && c <= 'z') {
            return c - 'a' + 26;
        } else if(c >= '0' && c <= '9') {
            return c - '0' + 52;
        }
        return -1; // 返回-1表示输入的字符无效
    }

    int countWords(int nodeIndex) {
        int result = 0;
        if (nodes[nodeIndex].is_end) {
            result++;
        }
        for (int i = 0; i < 62; i++) {
            if (nodes[nodeIndex].children[i] != -1) {
                result += countWords(nodes[nodeIndex].children[i]);
            }
        }
        return result;
    }

    void collectWords(int nodeIndex, const string& prefix, vector<string>& matches) {
        if (nodes[nodeIndex].is_end) {
            matches.push_back(prefix);
        }
        for (int i = 0; i < 62; i++) {
            if (nodes[nodeIndex].children[i] != -1) {
                char c = i < 26 ? 'A' + i : (i < 52 ? 'a' + i - 26 : '0' + i - 52);
                collectWords(nodes[nodeIndex].children[i], prefix + c, matches);
            }
        }
    }
};

int main() {
    ios::sync_with_stdio(false);

    int t,n,q,l;

    Trie trie;
    string s;
    cin >> t;
    for(int i = 0;i < t;i++) {
        cin >> n >> q;
        for(int j = 0;j < n;j++) {
            cin >> s;
            trie.insert(s);
        }
        for(int k = 0;k < q;k++) {
            cin >> s;
            cout << trie.prefixCount(s) << '\n';
        }
    }

    return 0;
}
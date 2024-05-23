// https://www.luogu.com.cn/problem/P8306
// 修了半天修不好，好歹过了一个AC了
// 1AC 5WA

#include <bits/stdc++.h>

using namespace std;

const size_t CHAR_SIZE = 63;// a-z A-Z 0-9

struct TrieNode {
    vector<TrieNode*> children;
    bool tail;
    TrieNode() : children(CHAR_SIZE, nullptr), tail(false) {}
    //~TrieNode() {
    //    for(const auto& node : children)
    //        delete node;
    //}
};

// Root node of the trie
TrieNode root;
// Number of nodes in the trie
int nodeCount = 0;

int get_index(char c) {
    if(c >= 'a' && c <= 'z') {
        return c - 'a';
    } else if(c >= 'A' && c <= 'Z') {
        return c - 'A' + 26;
    } else if(c >= '0' && c <= '9') {
        return c - '0' + 52;
    }
    return -1; // 返回-1表示输入的字符无效
}

void insert(TrieNode* node, const string& word) {
    for (const auto& ch : word) {
        int index = get_index(ch);
        if (!node->children[index]) {
            node->children[index] = new TrieNode();
            nodeCount++;
        }
        node = node->children[index];
    }
    node->tail = true;
}

bool search(TrieNode* node, const string& word) {
    for (const auto& ch : word) {
        int index = get_index(ch);
        if (!node->children[index]) {
            return false;
        }
        node = node->children[index];
    }
    // Return true if the last node is marked as end of word
    return node->tail;
}

int count(TrieNode* node) {
    if(!node)
        return 0;
    if(!node->children.empty()) {
        for(const auto& child : node->children) {
            if(child)
                return child->tail + count(child);
        }
    }
    return 1;
}

int count(TrieNode* node, const string& word) {
    for (const auto& ch : word) {
        int index = get_index(ch);
        if (!node->children[index]) {
            return 0;
        }
        node = node->children[index];
    }
    // Only count the node if it is the end of a word
    return node->tail ? count(node) : 0;
}

int main() {
    ios::sync_with_stdio(false);

    int t,n,q,l;
    string s;
    cin >> t;
    while(t-- > 0) {
        cin >> n >> q;
        while(n-- > 0) {
            cin >> s;
            insert(&root,s);
        }
        while(q-- > 0) {
            cin >> s;
            cout << count(&root,s) << '\n';
        }
    }

    return 0;
}
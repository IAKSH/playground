// https://www.luogu.com.cn/problem/P8306
// 怎么还是炸了？？
// 暂存

#include <bits/stdc++.h>

using namespace std;

// 字典树在查找上只比哈希表节约一些内存吧，除此之外啥也不是
// 但是字典树能做前缀匹配，哈希表不能

// 支持A-Z a-z 0-9的字典树
// 因为是打比赛用的，所以没有delete
class Trie {
private:
    struct TrieNode {
        TrieNode* children[62];
        bool isEndOfWord;
        TrieNode() {
            isEndOfWord = false;
            for (int i = 0; i < 62; i++)
                children[i] = nullptr;
        }
    };

    int charToInt(char c) {
        if(c >= 'A' && c <= 'Z')
            return c - 'A';
        if(c >= 'a' && c <= 'z')
            return c - 'a' + 26;
        if(c >= '0' && c <= '9')
            return c - '0' + 52;
        return -1;
    }

    int countWords(TrieNode* node) {
        if (!node)
            return 0;
        int count = 0;
        if (node->isEndOfWord)
            count++;
        for (int i = 0; i < 62; i++)
            if (node->children[i])
                count += countWords(node->children[i]);
        return count;
    }

    char intToChar(int i) {
        if(i >= 0 && i < 26)
            return 'A' + i;
        if(i >= 26 && i < 52)
            return 'a' + (i - 26);
        if(i >= 52 && i < 62)
            return '0' + (i - 52);
        return ' ';
    }

    void findAllWords(TrieNode* node, string &match, vector<string> &matches) {
        if (node->isEndOfWord)
            matches.push_back(match);
        for (int i = 0; i < 62; i++) {
            if (node->children[i]) {
                match.push_back(intToChar(i));
                findAllWords(node->children[i], match, matches);
                match.pop_back();
            }
        }
    }

public:
    TrieNode* root;

    Trie() {
        root = new TrieNode();
    }

    void insert(const string &word) {
        TrieNode* node = root;
        for (char c : word) {
            int index = charToInt(c);
            if (!node->children[index])
                node->children[index] = new TrieNode();
            node = node->children[index];
        }
        node->isEndOfWord = true;
    }

    bool search(const string &word) {
        TrieNode* node = root;
        for (char c : word) {
            int index = charToInt(c);
            if (!node->children[index])
                return false;
            node = node->children[index];
        }
        return node != nullptr && node->isEndOfWord;
    }

    int prefixCount(const string &prefix) {
        TrieNode* node = root;
        for (char c : prefix) {
            int index = charToInt(c);
            if (!node->children[index])
                return 0;
            node = node->children[index];
        }
        return countWords(node);
    }

    vector<string> prefixMatch(const string &prefix) {
        vector<string> matches;
        TrieNode* node = root;
        string match;
        for (char c : prefix) {
            int index = charToInt(c);
            if (!node->children[index])
                return matches;
            node = node->children[index];
            match.push_back(c);
        }
        findAllWords(node, match, matches);
        return matches;
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
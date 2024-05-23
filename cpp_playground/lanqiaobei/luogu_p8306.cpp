// https://www.luogu.com.cn/problem/P8306
// 全RE，不知道为什么，暂存

#include <bits/stdc++.h>

using namespace std;

#ifdef MINE_TRIE

struct Node {
    char c;
    bool tail;
    vector<Node*> sons;// a-z A-Z

    ~Node() {
        for(const auto& n : sons)
            delete n;
    }
};

struct Trie {
public:
    vector<Node> nodes;

    void insert(const string& s) {
        insert(s.begin(),s.end());
    }

    void insert(string::const_iterator str_begin,string::const_iterator str_end) {
        // only in first layer
        bool tail = (str_end - str_begin == 1);
        for(auto it = str_begin;it != str_end;it++) {
            auto n = find_if(nodes.begin(),nodes.end(),[&](const Node& node){return node.c == *it;});
            if(n == nodes.end())
                nodes.emplace_back(new Node{*it,tail});
            else {
                if(tail) {
                    n->tail = true;
                    return;
                }
                insert(str_begin + 1,str_end,*n);
            }
        }
    }

    int count(const string& s) {
        return count(s.begin(),s.end());
    }

    int count(string::const_iterator str_begin,string::const_iterator str_end) {
        for(auto it = str_begin;it != str_end;it++) {
            auto n = find_if(nodes.begin(),nodes.end(),[&](const Node& node){return node.c == *it;});
            if(n == nodes.end())
                break;
            else {
                if(str_end - str_begin == 1)
                    return 1;
                return 1 + count(str_begin + 1,str_end,*n);
            }
        }
        return 0;
    }

private:
    void insert(const string::const_iterator& str_begin,const string::const_iterator& str_end,Node& node) {
        bool tail = (str_end - str_begin == 1);
        for(auto it = str_begin;it != str_end;it++) {
            auto n = find_if(node.sons.begin(),node.sons.end(),[&](const Node& node){return node.c == *it;});
            if(n == node.sons.end()) {
                node.sons.emplace_back(new Node{*it},tail);
            }
            else {
                if(tail) {
                    (*n)->tail = true;
                    return;
                }
                insert(str_begin + 1,str_end,**n);
            }
        }
    }

    int count(const string::const_iterator& str_begin,const string::const_iterator& str_end,Node& node) {
        for(auto it = str_begin;it != str_end;it++) {
            auto n = find_if(node.sons.begin(),node.sons.end(),[&](const Node& node){return node.c == *it;});
            if(n == node.sons.end())
                break;
            else {
                if(str_end - str_begin == 1)
                    return 1;
                return 1 + count(str_begin + 1,str_end,**n);
            }
        }
        return 0;
    }
};

#else

const size_t CHAR_SIZE = 52;// a-z A-Z 0-9

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
    if(c > '0' && c < '9') {
        return c - '0' + 52;
    }
    return c - 'a';
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
            return false;
        }
        node = node->children[index];
    }
    // search all children
    return count(node);
}

#endif

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
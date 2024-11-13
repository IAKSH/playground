// https://leetcode.cn/problems/implement-trie-prefix-tree/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Trie {
public:
    Trie() {}
    
    void insert(string word) {
        vector<Node>* v = &nodes;
        for(int i = 0;i < word.size();i++) {
            auto it = find_node(v,word[i]);
            if(it != v->end()) {
                if(i == word.size() - 1)
                    it->is_terminal = true;
                v = &it->nexts;
            }
            else {
                v->emplace_back(Node{word[i],i == word.size() - 1});
                v = &v->back().nexts;
            }
        }
    }
    
    bool search(string word) {
        vector<Node>* v = &nodes;
        for(int i = 0;i < word.size();i++) {
            auto it = find_node(v,word[i]);
            if(it != v->end()) {
                if(i == word.size() - 1)
                    return it->is_terminal;
                v = &it->nexts;
            }
            else
                return false;
        }
        return false;
    }
    
    bool startsWith(string prefix) {
        vector<Node>* v = &nodes;
        for(int i = 0;i < prefix.size();i++) {
            auto it = find_node(v,prefix[i]);
            if(it != v->end()) {
                if(i == prefix.size() - 1)
                    return true;
                v = &it->nexts;
            }
            else
                return false;
        }
        return false;
    }

private:
    struct Node {
        char c;
        bool is_terminal;
        vector<Node> nexts;
    };

    vector<Node> nodes;

    vector<Node>::iterator find_node(vector<Node>* v,char c) {
        auto it = v->begin();
        while(it != v->end() && it->c != c)
            ++it;
        return it;
    }
};

int main() {
    cout << "i don't wanna write a test\n";
    return 0;
}
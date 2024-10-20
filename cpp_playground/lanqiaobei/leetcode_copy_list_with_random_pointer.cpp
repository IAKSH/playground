// https://leetcode.cn/problems/copy-list-with-random-pointer/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Node {
public:
    int val;
    Node* next;
    Node* random;
    
    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};

class Solution {
public:
    Node* copyRandomList(Node* head) {
        Node* node = head;
        Node* new_head = nullptr;
        Node* new_node;
        unordered_map<Node*,Node*> mem; 
        while(node) {
            if(!new_head) {
                new_head = new Node(head->val);
                new_node = new_head;
                mem[head] = new_head;
            }
            else {
                new_node->next = new Node(node->val);
                new_node = new_node->next;
                mem[node] = new_node;
            }
            node = node->next;
        }

        node = head;
        new_node = new_head;
        while(new_node) {
            new_node->random = mem[node->random];
            node = node->next;
            new_node = new_node->next;
        }

        return new_head;
    }
};

int main() {
    cout << "someone didn't write a test due to the lack of motivation\n";
    return 0;
}
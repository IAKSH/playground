// https://leetcode.cn/problems/linked-list-cycle-ii/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        unordered_set<ListNode*> mem;
        ListNode* node = head;
        while(node) {
            if(mem.count(node))
                return node;
            mem.emplace(node);
            node = node->next;
        }
        return NULL;
    }
};

int main() {
    // 略
    return 0;
}
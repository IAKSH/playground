// https://leetcode.cn/problems/remove-nth-node-from-end-of-list/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        deque<ListNode*> dq;
        ListNode* node = head;
        while(node) {
            dq.emplace_back(node);
            if(dq.size() > n + 1)
                dq.pop_front();
            node = node->next;
        }

        if(dq.size() == n) {
            delete dq[0];
            head = (n == 1 ? nullptr : dq[1]);
        }
        else {
            dq.emplace_back(nullptr);
            delete dq[1];
            dq[0]->next = dq[2];
        }
        
        return head;
    }
};

int main() {
    ListNode* head = new ListNode;
    head->val = 1;
    head->next = new ListNode;
    head->next->val = 2;
    head->next->next = NULL;

    head = Solution().removeNthFromEnd(head,1);

    return 0;
}
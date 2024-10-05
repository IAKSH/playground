// https://leetcode.cn/problems/swap-nodes-in-pairs/description/?envType=study-plan-v2&envId=top-100-liked

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
    ListNode* swapPairs(ListNode* head) {
        if(!head || !head->next)
            return head;

        ListNode *a,*b,*c,*d;
        b = head;
        c = b->next;
        d = c->next;

        head = c;
        c->next = b;
        b->next = d;

        while(c) {
            swap(b,c);
            for(int i = 0;i < 2;i++) {
                if(!d)
                    return head;
                a = b;
                b = c;
                c = d;
                d = d->next;
            }
            a->next = c;
            c->next = b;
            b->next = d;
        }

        return head;
    }
};

int main() {
    ListNode* head = new ListNode;
    head->next = new ListNode;
    head->next->next = new ListNode;
    head->next->next->next = new ListNode;
    head->next->next->next->next = nullptr;

    head->val = 1;
    head->next->val = 2;
    head->next->next->val = 3;
    head->next->next->next->val = 4;

    head = Solution().swapPairs(head);

    while(head) {
        cout << head->val << ' ';
        head = head->next;
    }
    cout << '\n';

    return 0;
}

/*
输入：head = [1,2,3,4]
输出：[2,1,4,3]
*/
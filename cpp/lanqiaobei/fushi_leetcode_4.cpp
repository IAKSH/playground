// https://leetcode.cn/problems/reverse-linked-list/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>
using namespace std;

// Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution1 {
public:
    ListNode* reverseList(ListNode* head) {
        l = head;
        recurse(head);
        return head;
    }

private:
    bool b = false;
    ListNode* l;
    void recurse(ListNode* r) {
        if(!r) return;
        recurse(r->next);
        if(b) 
            return;
        if(r == l || r->next == l) {
            b = true;
            return;
        }
        swap(r->val,l->val);
        l = l->next;
    }
};

class Solution2 {
private:
    ListNode* new_head = nullptr;
    void recurse(ListNode* r,ListNode* pre_r) {
        if(r->next) recurse(r->next,r);
        else new_head = r;
        r->next = pre_r;
    }
public:
    ListNode* reverseList(ListNode* head) {
        if(!head) return nullptr;
        recurse(head,nullptr);
        return new_head;
    }
};

class Solution3 {
private:
    ListNode* last = nullptr;
public:
    ListNode* reverseList(ListNode* head) {
        if(!head) return last;
        auto real_next = head->next;
        head->next = last;
        last = head;
        return reverseList(real_next);
    }
};

class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode *last = nullptr, *next = nullptr;
        while(head) {
            next = head->next;
            head->next = last;
            last = head;
            head = next;
        }
        return last;
    }
};

int main() {
    ListNode n2(2);
    ListNode n1(1,&n2);
    
    auto p = Solution().reverseList(&n1);
    while(p) {
        cout << p->val << ',';
        p = p->next;
    }
    cout << '\b' << endl;
    return 0;
}
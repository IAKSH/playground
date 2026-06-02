// https://leetcode.cn/problems/linked-list-cycle/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>
using namespace std;

// Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

class Solution1 {
public:
    bool hasCycle(ListNode *head) {
        unordered_set<ListNode*> map;
        while(head) {
            if(map.find(head) != map.end())
                return true;
            else {
                map.emplace(head);
                head = head->next;
            }
        };
        return false;
    }
};

// 快慢指针
class Solution {
public:
    bool hasCycle(ListNode *head) {
        if(!head)
            return false;
        ListNode *l = head,*r = head->next;
        while(l || r) {
            if(l == r)
                return true;
            for(int i = 0;i < 2;i++) {
                if(r)
                    r = r->next;
            }
            if(l)
                l = l->next;
        }
        return false;
    }
};

int main() {

}
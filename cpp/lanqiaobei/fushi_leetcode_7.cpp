// https://leetcode.cn/problems/merge-two-sorted-lists/?envType=study-plan-v2&envId=top-100-liked

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

class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode *out_last = nullptr,*out_begin = nullptr;

        auto foo = [&](ListNode*& cur){
            if(out_last)
                out_last->next = cur;
            else
                out_begin = cur;
            out_last = cur;
            cur = cur->next;
        };

        while(list1 || list2) {
            if(list1 && list2) {
                if(list1->val < list2->val) {
                    foo(list1);
                }
                else {
                    foo(list2);
                }
            }
            else if(list1) {
                foo(list1);
            }
            else {
                foo(list2);
            }
        }
        
        return out_begin;
    }
};

int main() {

}
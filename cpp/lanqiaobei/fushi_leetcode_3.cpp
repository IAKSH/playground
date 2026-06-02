// https://leetcode.cn/problems/intersection-of-two-linked-lists/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>
using namespace std;


// Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

class SolutionHash {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        unordered_set<ListNode*> set;
        ListNode* p = headA;
        while(p) {
            set.emplace(p);
            p = p->next;
        }
        p = headB;
        while(p) {
            if(set.find(p) != set.end()) {
                return p;
            }
            p = p->next;
        }
        return nullptr;
    }
};

// 双指针，通过换路来抵销了一开始的距离差，达成了同步。
// （让原本路径更长的在第二遍去走路径更短的，让原本路径更短的在第二遍去走路径更长的）
// （A：n + delta + n - delta ; B: n - delta + n + delta）
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if(!headA || !headB)
            return nullptr;
        auto ptr_a = headA,ptr_b = headB;
        while(ptr_a != ptr_b) {
            ptr_a = ptr_a ? ptr_a->next : headB;
            ptr_b = ptr_b ? ptr_b->next : headA;
        }
        return ptr_a;
    }
};

int main() {

}
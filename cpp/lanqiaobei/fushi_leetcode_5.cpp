// https://leetcode.cn/problems/palindrome-linked-list/description/?envType=study-plan-v2&envId=top-100-liked

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
    bool isPalindrome(ListNode* head) {
        deque<int> dq;
        while(head) {
            dq.emplace_back(head->val);
            head = head->next;
        }
        while(dq.size() > 1) {
            if(dq.back() == dq.front()) {
                dq.pop_back();
                dq.pop_front();
            }
            else
                return false;
        }
        return true;
    }
};

class Solution {
public:
    bool isPalindrome(ListNode* head) {
        vector<int> v;
        while(head) {
            v.emplace_back(head->val);
            head = head->next;
        }
        int l = 0,r = v.size() - 1;
        while(l < r) {
            if(v[l] == v[r]) {
                ++l;
                --r;
            }
            else
                return false;
        }
        return true;
    }
};



int main() {
    //ListNode n5(1);
    //ListNode n4(2,&n5);
    //ListNode n3(0,&n4);
    //ListNode n2(2,&n3);
    //ListNode n1(1,&n2);

    ListNode n4(1);
    ListNode n3(2,&n4);
    ListNode n2(2,&n3);
    ListNode n1(1,&n2);

    cout << (Solution().isPalindrome(&n1) ? "true" : "false") << endl;
    return 0;
}
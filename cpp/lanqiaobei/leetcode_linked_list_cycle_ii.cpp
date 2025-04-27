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
        ListNode *f,*s;
        int i;
        f = s = head;

        while(f && f->next) {
            f = f->next->next;
            s = s->next;
            if(s == f) {
                f = head;
                while(f != s) {
                    f = f->next;
                    s = s->next;
                }
                return f;
            }
        }

        return NULL;
    }
};

int main() {
    // 略
    return 0;
}
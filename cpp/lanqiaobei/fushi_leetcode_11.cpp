// https://leetcode.cn/problems/symmetric-tree/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>
using namespace std;

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};
 
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        if(!root || (!root->left && !root->right))
            return true;
        else if(!root->left || !root->right)
            return false;
        
        stack<TreeNode*> st1,st2;
        st1.emplace(root->left);
        st2.emplace(root->right);

        TreeNode *l,*r;

        while(!st1.empty() && !st2.empty()) {
            l = st1.top();
            r = st2.top();
            if(l->val != r->val)
                return false;
            st1.pop();
            st2.pop();

            if(l->left && r->right) {
                st1.emplace(l->left);
                st2.emplace(r->right);
            }
            else if(l->left || r->right)
                return false;

            if(l->right && r->left) {
                st1.emplace(l->right);
                st2.emplace(r->left);
            }
            else if(l->right || r->left)
                return false;
        }

        return st1.empty() && st2.empty();
    }
};

// [2,3,3,4,null,5,4]
//    2
//  3   3
// 4 x 5 4

int main() {

}
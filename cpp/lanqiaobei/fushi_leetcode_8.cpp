// https://leetcode.cn/problems/binary-tree-inorder-traversal/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>
using namespace std;

//Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class Solution {
private:
    vector<int> v;
    void recurse(TreeNode* root) {
        if(root->left) recurse(root->left);
        v.emplace_back(root->val);
        if(root->right) recurse(root->right);
    }

public:
    vector<int> inorderTraversal(TreeNode* root) {
        if(root) recurse(root);
        return v;
    }
};

int main() {

}
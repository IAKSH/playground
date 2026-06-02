// https://leetcode.cn/problems/invert-binary-tree/description/?envType=study-plan-v2&envId=top-100-liked

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

class Solution1 {
public:
    TreeNode* invertTree(TreeNode* root) {
        if(root) {
            if(root->left && root->right) {
                swap(root->left,root->right);
                invertTree(root->left);
                invertTree(root->right);
            }
            else if(root->left) {
                root->right = root->left;
                root->left = nullptr;
                invertTree(root->right);
            }
            else {
                root->left = root->right;
                root->right = nullptr;
                invertTree(root->left);
            }
        }
        return root;
    }
};

class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if(!root) return nullptr;
        TreeNode* node = root->left;
        root->left = invertTree(root->right);
        root->right = invertTree(node);
        return root;
    }
};

int main() {

}
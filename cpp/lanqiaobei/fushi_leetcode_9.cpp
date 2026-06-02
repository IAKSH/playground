// https://leetcode.cn/problems/maximum-depth-of-binary-tree/description/?envType=study-plan-v2&envId=top-100-liked

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
private:
    int dep = 0;
    int depest = 0;
public:
    int maxDepth(TreeNode* root) {
        if(root) {
            ++dep;
            if(root->left)
                maxDepth(root->left);
            if(root->right)
                maxDepth(root->right);
            if(!root->left && !root->right) {
                depest = max(depest,dep);
            }
        }
        --dep;
        return depest;
    }
};

class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(root)
            return 1 + max(maxDepth(root->left),maxDepth(root->right));
        return 0;
    }
};

int main() {

}
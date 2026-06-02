// https://leetcode.cn/problems/diameter-of-binary-tree/?envType=study-plan-v2&envId=top-100-liked

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
    int depth(TreeNode* root) {
        return root ? 1 + max(depth(root->left),depth(root->right)) : 0;
    }
public:
    int diameterOfBinaryTree(TreeNode* root) {
        int d_l = 0,d_r = 0,dep_l = 0,dep_r = 0;
        if(root->left) {
            dep_l = depth(root->left);
            d_l = diameterOfBinaryTree(root->left);
        }
        if(root->right) {
            dep_r = depth(root->right);
            d_r = diameterOfBinaryTree(root->right);
        }
        return max(dep_l + dep_r,max(d_l,d_r));
    }
};

class Solution {
private:
    int d_max = 0;
    int dfs(TreeNode* root) {
        // 后序遍历，在向上合并时检查并保留最大d
        int dep_l = 0,dep_r = 0;
        if(root->left) dep_l = dfs(root->left);
        if(root->right) dep_r = dfs(root->right);
        d_max = max(d_max,dep_l + dep_r);
        return 1 + max(dep_l,dep_r);
    }
public:
    int diameterOfBinaryTree(TreeNode* root) {
        dfs(root);
        return d_max;
    }
};

int main() {
    
}
// https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

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
    void flatten(TreeNode* root) {
        if(!root)
            return;
        TreeNode* new_root = new TreeNode(root->val);
        auto result_node = new_root;
        bool b = root->left;
        if(b) {
            dfs(result_node,root->left);
        }
        if(root->right) {
            if(b)
                result_node = result_node->right;
            dfs(result_node,root->right);
        }   
        *root = *new_root;
    }

private:
    void dfs(TreeNode*& result_node,TreeNode* node) {
        result_node->right = new TreeNode(node->val);
        result_node->left = nullptr;
        if(node->left) {
            result_node = result_node->right;
            dfs(result_node,node->left);
        }
        if(node->right) {
            result_node = result_node->right;
            dfs(result_node,node->right);
        }
    }
};

int main() {
    cout << "I don't wanna to write a test\n";
    return 0;
}
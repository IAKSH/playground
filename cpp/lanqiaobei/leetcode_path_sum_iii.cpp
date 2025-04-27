// https://leetcode.cn/problems/path-sum-iii/?envType=study-plan-v2&envId=top-100-liked

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
    int pathSum(TreeNode* root, int targetSum) {
        target = targetSum;
        cnt = 0;
        if(!root)
            return cnt;
        prefix_dfs(root,0);
        dfs1(root);
        return cnt;
    }

private:
    void prefix_dfs(TreeNode* node,TreeNode* root) {
        mem[node] = node->val + mem[root];
        if(node->left)
            prefix_dfs(node->left,node);
        if(node->right)
            prefix_dfs(node->right,node);
    }

    void dfs1(TreeNode* node) {
        if(mem[node] == target)
            ++cnt;
        if(node->left) {
            dfs2(node,node->left);
            dfs1(node->left);
        }
        if(node->right) {
            dfs2(node,node->right);
            dfs1(node->right);
        }
    }

    void dfs2(TreeNode* start_node,TreeNode* node) {
        if(mem[node] - mem[start_node] == target)
            ++cnt;
        if(node->left)
            dfs2(start_node,node->left);
        if(node->right)
            dfs2(start_node,node->right);
    }

    int target;
    int cnt;

    unordered_map<TreeNode*,long long> mem;
};

int main() {

}
// https://leetcode.cn/problems/diameter-of-binary-tree/description/?envType=study-plan-v2&envId=top-100-liked

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
    int diameterOfBinaryTree(TreeNode* root) {
        int max_depth = 0;
        max_depth = max(max_depth,dfs(max_depth,root) - 1);
        return max_depth;
    }

private:
    int dfs(int& max_depth,TreeNode* node) {
        int l = 0,r = 0;
        if(node->left)
            l = dfs(max_depth,node->left);
        if(node->right)
            r = dfs(max_depth,node->right);
        max_depth = max(max_depth,l + r);
        return max(l,r) + 1;
    }
};

int main() {
    cout << "no need for test\n";
    return 0;
}
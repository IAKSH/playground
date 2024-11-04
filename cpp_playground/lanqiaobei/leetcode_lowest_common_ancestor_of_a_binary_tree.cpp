// https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        cnt = 0;
        result = nullptr;
        val1 = p->val;
        val2 = q->val;
        dfs(root);
        return result;
    }

private:
    bool dfs(TreeNode* node) {
        if(result || cnt == 2)
            return false;

        bool b1,b2;
        b1 = b2 = false;
        if(node->left)
            b1 = dfs(node->left);
        if(node->right)
            b2 = dfs(node->right);
    
        if(node->val == val1 || node->val == val2) {
            ++cnt;
            if(b1 || b2)
                result = node;
            return true;
        }
        else {
            if(b1 && b2)
                result = node;
            return (b1 || b2);
        }
    }

    int val1,val2;
    int cnt;
    TreeNode* result;
};

int main() {
    cout << "no test for you!\n";
    return 0;
}
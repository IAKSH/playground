// https://leetcode.cn/problems/symmetric-tree/description/?envType=study-plan-v2&envId=top-100-liked

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
    bool isSymmetric(TreeNode* root) {
        if(!root)
            return true;
        vector<int> v1,v2;
        dfs1(v1,root->left);
        dfs2(v2,root->right);

        return v1 == v2;
    }

private:
    void dfs1(vector<int>& v,TreeNode* node) {
        if(node) {
            v.emplace_back(node->val);
            dfs1(v,node->left);
            dfs1(v,node->right);
        }
        else
            v.emplace_back(-1);
    }

    void dfs2(vector<int>& v,TreeNode* node) {
        if(node) {
            v.emplace_back(node->val);
            dfs2(v,node->right);
            dfs2(v,node->left);
        }
        else
            v.emplace_back(-1);
    }
};

int main() {
    cout << "no need for test\n";
    return 0;
}
// https://leetcode.cn/problems/kth-smallest-element-in-a-bst/description/?envType=study-plan-v2&envId=top-100-liked

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
    int kthSmallest(TreeNode* root, int k) {
        vector<int> v;
        dfs(v,root);
        return v[k - 1];
    }

private:
    void dfs(vector<int>& v, TreeNode* node) {
        if(node->left)
            dfs(v,node->left);
        v.emplace_back(node->val);
        if(node->right)
            dfs(v,node->right);
    }
};

int main() {
    cout << "no need for test\n";
    return 0;
}
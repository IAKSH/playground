// https://leetcode.cn/problems/validate-binary-search-tree/?envType=study-plan-v2&envId=top-100-liked

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
    bool isValidBST(TreeNode* root) {
        vector<int> v;
        dfs(v,root);
        for(int i = 1;i < v.size();i++) {
            if(v[i - 1] >= v[i])
                return false;
        }
        return true;
    }

private:
    void dfs(vector<int>& v,TreeNode* node) {
        if(node->left)
            dfs(v,node->left);
        v.emplace_back(node->val);
        if(node->right)
            dfs(v,node->right);
    }
};

int main() {
    TreeNode* root = new TreeNode(2,
        new TreeNode(1),new TreeNode(3)
    );
    cout << (Solution().isValidBST(root) ? "true" : "false") << '\n';
    return 0;
}
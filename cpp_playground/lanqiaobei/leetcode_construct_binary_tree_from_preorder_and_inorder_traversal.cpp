// https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/?envType=study-plan-v2&envId=top-100-liked

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
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int pre_offset = 0;
        return recurse(preorder,inorder,pre_offset,0,preorder.size());
    }

private:
    TreeNode* recurse(const vector<int>& preorder, const vector<int>& inorder,int& pre_offset,int in_l,int in_r) {
        TreeNode* root = new TreeNode(preorder[pre_offset]);
        int i = in_l;
        for(;i < in_r && inorder[i] != preorder[pre_offset];i++);
        if(in_l != i) {
            ++pre_offset;
            root->left = recurse(preorder,inorder,pre_offset,in_l,i);
        }
        if(in_r > i + 1) {
            ++pre_offset;
            root->right = recurse(preorder,inorder,pre_offset,i + 1,in_r);
        }
        return root;
    }
};

void check_pre(vector<int>& v,TreeNode* node) {
    v.emplace_back(node->val);
    if(node->left)
        check_pre(v,node->left);
    if(node->right)
        check_pre(v,node->right);
}

void check_in(vector<int>& v,TreeNode* node) {
    if(node->left)
        check_in(v,node->left);
    v.emplace_back(node->val);
    if(node->right)
        check_in(v,node->right);
}

int main() {
    vector<int> preorder{3,9,20,15,7};
    vector<int> inorder{9,3,15,20,7};

    vector<int> res_pre,res_in;
    TreeNode* root = Solution().buildTree(preorder,inorder);
    check_pre(res_pre,root);
    check_in(res_in,root);

    bool b[2]{res_pre == preorder,res_in == inorder};
    if(b[0] && b[1])
        cout << "passed\n";
    else {
        if(!b[0])
            cout << "preorder missmatch\n";
        if(!b[1])
            cout << "inorder missmatch\n";
    }

    return 0;
}
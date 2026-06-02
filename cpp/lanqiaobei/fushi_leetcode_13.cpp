// https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/?envType=study-plan-v2&envId=top-100-liked

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

class Solution {
private:
    // 将BST拍扁（中序遍历），发现规律：有序，每个子树的root对应在该子树在数组中所在区间的上下界中央位置（上下界下标平均值向下取整）
    void recurse(TreeNode* n,vector<int>& v,int l,int r) {
        int mid = (l + r) / 2;
        n->val = v[mid];
        if(l < mid) {
            n->left = new TreeNode;
            recurse(n->left,v,l,mid);   
        }
        if(r - 1 > mid) {
            n->right = new TreeNode;
            recurse(n->right,v,mid + 1,r);
        }
    }

public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        TreeNode* n = new TreeNode;
        recurse(n,nums,0,nums.size());
        return n;
    }
};

int main() {

}
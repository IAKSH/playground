// https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/?envType=study-plan-v2&envId=top-100-liked

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
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        TreeNode* root = new TreeNode;
        recursion(nums,root,0,nums.size());
        return root;
    }

private:
    void recursion(const vector<int>& nums,TreeNode* node,int l,int r) {
        int mid = (l + r) / 2; 
        node->val = nums[mid];
        if(l + 1 < r) {
            if(mid > l) {
                node->left = new TreeNode;
                recursion(nums,node->left,l,mid);
            }
            if(mid + 1 < r) {
                node->right = new TreeNode;
                recursion(nums,node->right,mid + 1,r);
            }
        }
    }
};

int main() {
    cout << "no need for test\n";
    return 0;
}
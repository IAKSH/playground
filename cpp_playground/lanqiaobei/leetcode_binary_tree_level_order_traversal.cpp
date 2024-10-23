// https://leetcode.cn/problems/binary-tree-level-order-traversal/description/?envType=study-plan-v2&envId=top-100-liked

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
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> v;
        deque<pair<TreeNode*, int>> nodes;
        int depth = 0;

        if(root)
            nodes.emplace_back(make_pair(root, depth));

        while(!nodes.empty()) {
            auto node_pair = nodes.front();
            nodes.pop_front();
            TreeNode* node = node_pair.first;
            depth = node_pair.second;

            if(v.size() <= depth)
                v.emplace_back(vector<int>());

            if(node) {
                v[depth].emplace_back(node->val);
                nodes.emplace_back(make_pair(node->left, depth + 1));
                nodes.emplace_back(make_pair(node->right, depth + 1));
            }
        }
        if(v.size() > 0)
            v.pop_back();
        return v;
    }
};


int main() {
    cout << "no need for test\n";
    return 0;
}
// https://leetcode.cn/problems/binary-tree-right-side-view/description/?envType=study-plan-v2&envId=top-100-liked

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
    vector<int> rightSideView(TreeNode* root) {
        vector<int> v;
        if(!root)
            return v;
        v.resize(1);
        deque<pair<TreeNode*,int>> dq;
        bfs(v,dq,root,0);
        return v;
    }

private:
    void bfs(vector<int>& v,deque<pair<TreeNode*,int>>& dq,TreeNode* node,int depth) {
        if(node->left)
            dq.emplace_back(make_pair(node->left,depth + 1));
        if(node->right)
            dq.emplace_back(make_pair(node->right,depth + 1));
        v[depth] = node->val;
        if(!dq.empty()) {
            auto next = dq.front();
            dq.pop_front();
            if(next.second > depth)
                v.emplace_back(0);
            bfs(v,dq,next.first,next.second);
        }
    }
};

int main() {
    cout << "no need for test\n";
    return 0;
}
#include <stdio.h>

typedef struct __BSTNode {
    struct __BSTNode* left;
    struct __BSTNode* right;
    int val;
} BSTNode;

/*
试编写一算法，求出指定结点（值为k）在给定的二叉排序树中所在的层数。
注意：给定结点一定在二叉排序树上。
*/

int search_depth_from_bst(BSTNode* node,int* depth,int val) {
    // 由于“给定结点一定在二叉排序树上”，不考虑NULL
    // 似乎return int可以直接拆了，就拿int* depth当结果
    // 但是我懒
    if(node->val > val) {
        ++(*depth);
        return search_depth_from_bst(node->left,depth,val);
    }
    else if(node->val < val) {
        ++(*depth);
        return search_depth_from_bst(node->right,depth,val);
    }
    else {
        return *depth;
    }
}

int main(void) {

}
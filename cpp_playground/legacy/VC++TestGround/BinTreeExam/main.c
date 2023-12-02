#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct __LinkedBinTreeNode {
	struct __LinkedBinTreeNode* left;
	struct __LinkedBinTreeNode* right;
	int val;
} LinkedBinTreeNode;

typedef LinkedBinTreeNode* LinkedBinTree;

LinkedBinTree binTreeCreate(void) {
	return NULL;
}

void binTreeAdd(LinkedBinTree* tree,int val) {
	if (!(*tree)) {
		(*tree) = (LinkedBinTree)malloc(sizeof(LinkedBinTree));
		(*tree)->left = NULL;
		(*tree)->right = NULL;
		(*tree)->val = val;
	}
	else if ((*tree)->val >= val)
		binTreeAdd(&((*tree)->left), val);
	else
		binTreeAdd(&((*tree)->right), val);
}

void binTreePreTrav(LinkedBinTree tree, void(*callback)(int val)) {
	if (tree) {
		callback(tree->val);
		binTreePreTrav(tree->left, callback);
		binTreePreTrav(tree->right, callback);
	}
}

void binTreeInTrav(LinkedBinTree tree, void(*callback)(int val)) {
	if (tree) {
		binTreeInTrav(tree->left, callback);
		callback(tree->val);
		binTreeInTrav(tree->right, callback);
	}
}

void binTreePostTrav(LinkedBinTree tree, void(*callback)(int val)) {
	if (tree) {
		binTreePostTrav(tree->left, callback);
		binTreePostTrav(tree->right, callback);
		callback(tree->val);
	}
}

static int nodes = 0;
static int currentDepth = 0;
static int depth = 0;
void countBinTreeDepthAndNodes(LinkedBinTree tree) {
	if (tree) {
		// 到达了新节点，记录
		++nodes;
		++currentDepth;
		// 尝试进入左右子树
		countBinTreeDepthAndNodes(tree->left);
		countBinTreeDepthAndNodes(tree->right);
		// 回退到上层，更新当前深度
		--currentDepth;
	}
	else {
		// 降到叶时，尝试记录深度
		if (currentDepth > depth)
			depth = currentDepth;
	}
}

// Depth: k
// Node: 2^k-1 
int binTreeIsFull(LinkedBinTree tree) {
	countBinTreeDepthAndNodes(tree);
	int n = pow(2, depth) - 1;
	return nodes == n;
}

void __printBinTreeNode(int val) {
	printf("%d ", val);
	puts("");
}

int main(void) {
	LinkedBinTree binTree = binTreeCreate();
	int total, input;
	scanf("%d", &total);
	for (int i = 0; i < total; i++) {
		scanf("%d", &input);
		binTreeAdd(&binTree, input);
	}

	puts("前序遍历:");
	binTreePreTrav(binTree, __printBinTreeNode);
	puts("中序遍历:");
	binTreeInTrav(binTree, __printBinTreeNode);
	puts("后序遍历:");
	binTreePostTrav(binTree, __printBinTreeNode);
	printf(binTreeIsFull(binTree) ? "是满二叉树" : "不是满二叉树");
}
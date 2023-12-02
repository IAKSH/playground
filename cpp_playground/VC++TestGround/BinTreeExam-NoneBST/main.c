#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 非BST
typedef struct __LinkedBinTreeNode {
	struct __LinkedBinTreeNode* left;
	struct __LinkedBinTreeNode* right;
	int val;
} LinkedBinTreeNode;

typedef LinkedBinTreeNode* LinkedBinTree;

LinkedBinTree binTreeCreate(void) {
	return NULL;
}

void binTreeUserAdd(LinkedBinTree* tree) {
	int input;
	while (1) {
		if (!(*tree)) {
			puts("抵达叶\n为叶赋值...");
			scanf("%d", &input);

			// 创建叶
			(*tree) = (LinkedBinTree)malloc(sizeof(LinkedBinTree));
			(*tree)->left = NULL;
			(*tree)->right = NULL;
			(*tree)->val = input;

			printf("选择移动方向(-1:左 1:右 x:上)...");
			scanf("%d", &input);
			if (input == -1)
				binTreeUserAdd(&((*tree)->left));
			else if (input == 1)
				binTreeUserAdd(&((*tree)->right));
			else
				break;
		}
		else {
			printf("抵达分支节点\n选择移动方向(-1:左 1:右 x:上)...");
			scanf("%d", &input);
			if (input == -1)
				binTreeUserAdd(&((*tree)->left));
			else if (input == 1)
				binTreeUserAdd(&((*tree)->right));
			else
				break;
		}
	}
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

// 这种算法实际上效率应该是不如暴力搜索的
// 因为暴力搜索可以在判断到任意最小树不满时，直接退出，返回false，而不再遍历后面的节点
// 而公式法则无论如何都需要进行一遍完整的遍历
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
int __binTreeIsFull(LinkedBinTree tree) {
	countBinTreeDepthAndNodes(tree);
	int n = pow(2, depth) - 1;
	return nodes == n;
}

// 不可用，因为子树不满和子树为空都会返回false，然后被判断为左右皆空的情况，令最小树（的上面的树）被判断为满
int binTreeIsFull(LinkedBinTree tree) {
	return tree && (binTreeIsFull(tree->left) == binTreeIsFull(tree->right));
}

/*
// 可以用队列实现广度优先搜索
// 没有递归，且很少遍历整个树
// TODO: ...
int binTreeIsFull_Fast(LinkedBinTree tree) {

}
*/

void __printBinTreeNode(int val) {
	printf("%d ", val);
	puts("");
}

int main(void) {
	LinkedBinTree binTree = binTreeCreate();
	binTreeUserAdd(&binTree);

	puts("\n---");
	countBinTreeDepthAndNodes(binTree);
	printf("树的深度 = %d\n", depth);
	printf("树的节点数 = %d\n", nodes);

	puts("前序遍历:");
	binTreePreTrav(binTree, __printBinTreeNode);
	puts("中序遍历:");
	binTreeInTrav(binTree, __printBinTreeNode);
	puts("后序遍历:");
	binTreePostTrav(binTree, __printBinTreeNode);
	printf(__binTreeIsFull(binTree) ? "是满二叉树" : "不是满二叉树");
}
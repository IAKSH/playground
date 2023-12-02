#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct __LinkStackNode{
	char c;
	struct __LinkStackNode* next;
} LinkStackNode;

void linkstack_init(LinkStackNode* linkstack) {
	linkstack->next = NULL;
}

void linkstack_push(LinkStackNode* linkstack,char c) {
	LinkStackNode* new_node = (LinkStackNode*)malloc(sizeof(LinkStackNode));
	new_node->c = c;
	new_node->next = linkstack->next;
	linkstack->next = new_node;
}

char linkstack_pop(LinkStackNode* linkstack) {
	LinkStackNode* top = linkstack->next;
	if (!top)
		fprintf(stderr, "链栈空\n");
	else {
		char c = top->c;
		linkstack->next = top->next;
		free(top);
		return c;
	}	
}

// 12321
// 1221

int main(int argc,char** argv) {
	if (argc != 2) {
		fprintf(stderr, "使用方式：ispalindrome [str]\n");
		return -1;
	}

	LinkStackNode linkstack;
	linkstack_init(&linkstack);

	char* input = argv[1];
	int len = strlen(input);

	int i;
	for (i = 0; i < len / 2; i++)
		linkstack_push(&linkstack, input[i]);

	i += len % 2;
	for (; i < len; i++) {
		if (input[i] != linkstack_pop(&linkstack)) {
			fprintf(stdout, "不是回文\n");
			return;
		}
	}

	fprintf(stdout, "是回文\n");
	return 0;
}

/*
int main()
{
	char input[512];
	int input_count;
	for (input_count = 0; 1; input_count++) {
		if (input_count == 512)
			fprintf(stderr, "to many input char");
		else {
			scanf("%c", input + input_count);
			if (input[input_count] == ' ' || input[input_count] == '\n')
				break;
		}
	}

	LinkStackNode linkstack;
	linkstack_init(&linkstack);

	int i;
	for (i = 0; i < input_count / 2; i++)
		linkstack_push(&linkstack, input[i]);

	i += input_count % 2;
	for (; i < input_count; i++) {
		if (input[i] != linkstack_pop(&linkstack)) {
			fprintf(stdout, "不是回文\n");
			return;
		}
	}

	fprintf(stdout, "是回文\n");
	return 0;
}
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_QUEUE_MAX_LEN 128
typedef struct __ArrayQueue {
	int data[ARRAY_QUEUE_MAX_LEN];
	int rear;
} ArrayQueue;

void arrayQueueInit(ArrayQueue* queue) {
	memset(queue->data, 0, ARRAY_QUEUE_MAX_LEN);
	queue->rear = 0;
}

void arrayQueuePush(ArrayQueue* queue, int val) {
	if (queue->rear + 1 == ARRAY_QUEUE_MAX_LEN) {
		puts("数组队列满，未添加");
		return;
	}
	queue->data[queue->rear++] = val;
}

int arrayQueuePop(ArrayQueue* queue) {
	if (queue->rear == 0) {
		puts("数组队列空，返回0");
		return 0;
	}
	else {
		int ret = queue->data[0];
		for (int i = 0; i < queue->rear - 1; i++)
			queue->data[i] = queue->data[i + 1];

		--queue->rear;
		return ret;
	}
}

int main(void) {
	ArrayQueue queue;
	arrayQueueInit(&queue);

	for (int i = 0; i < 129; i++)
		arrayQueuePush(&queue, i);

	for (int i = 0; i < 130; i++)
		printf("%d ", arrayQueuePop(&queue));

	return 0;
}
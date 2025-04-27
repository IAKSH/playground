#include <stdio.h>
#include <stdlib.h>

#define MAXSIZE 100

typedef struct {
    int data[MAXSIZE];
    int front;
    int rear;
    int tag;
} Queue;

void queue_init(Queue* q) {
    q->front = 0;
    q->rear = 0;
    q->tag = 0;
}

int queue_push(Queue* q, int e) {
    if (q->front == q->rear && q->tag == 1)
        return -1;
    else {
        q->data[q->rear] = e;
        q->rear = (q->rear + 1) % MAXSIZE;
        q->tag = 1;
        return 0;
    }
}

int queue_pop(Queue* q, int* e) {
    if (q->front == q->rear && q->tag == 0)
        return -1;
    else {
        *e = q->data[q->front];
        q->front = (q->front + 1) % MAXSIZE;
        q->tag = 0;
        return 0;
    }
}
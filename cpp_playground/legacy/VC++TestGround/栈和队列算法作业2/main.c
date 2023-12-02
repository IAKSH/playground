#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node* next;
} Node;

typedef Node* Queue;

Queue queue_init() {
    Node* node = (Node*)malloc(sizeof(Node));
    node->next = node;
    return node;
}

void queue_push(Queue* q, int e) {
    Node* node = (Node*)malloc(sizeof(Node));
    node->data = e;
    node->next = (*q)->next;
    (*q)->next = node;
    *q = node;
}

int queue_pop(Queue* q, int* e) {
    if (*q == (*q)->next)
        return -1;
    else {
        Node* head = (*q)->next; // 头结点
        Node* first = head->next; // 第一个结点
        *e = first->data;
        head->next = first->next;
        if (first == *q)
            *q = head;

        free(first);
        return 0;
    }
}
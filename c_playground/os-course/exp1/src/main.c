#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TIME_SLICE 1  // 时间片大小

typedef enum {
    R,W,B
} State;

typedef struct PCB {
    char name[20];
    State state;
    int used_time;     // 已用CPU时间（需运行的时间片个数）
    int arrive_time;   // 到达时间
    int cpu_time;      // 所需CPU总时间
    struct PCB* next;
} PCB;

typedef struct QueueNode {
    PCB pcb;
    struct QueueNode *next;
} QueueNode;

typedef struct Queue {
    QueueNode *head, *tail;
} Queue;

void queue_init(Queue *queue) {
    queue->head = queue->tail = NULL;
}

void queue_push_back(Queue *q, PCB pcb) {
    QueueNode *newNode = (QueueNode *)malloc(sizeof(QueueNode));
    newNode->pcb = pcb;
    newNode->next = NULL;
    if (q->tail == NULL) {
        q->head = q->tail = newNode;
    } else {
        q->tail->next = newNode;
        q->tail = newNode;
    }
}

PCB queue_pop_front(Queue *q) {
    if (q->head == NULL) {
        fprintf(stderr,"trying to pop from an empty queue\n");
        abort();
    }
    QueueNode *temp = q->head;
    PCB pcb = temp->pcb;
    q->head = q->head->next;
    if (q->head == NULL) {
        q->tail = NULL;
    }
    free(temp);
    return pcb;
}

int counter = 5;

void dispatch(Queue *ready_q, Queue *block_q, Queue *run_q) {
    int count = 0;
    while (ready_q->head != NULL || run_q->head != NULL) {
        if (run_q->head == NULL && ready_q->head != NULL) {
            PCB pcb = queue_pop_front(ready_q);
            pcb.state = R;
            queue_push_back(run_q, pcb);
        }

        PCB *running = &run_q->head->pcb;
        printf("Running process: %s\n", running->name);
        running->used_time += TIME_SLICE;

        if (running->used_time >= running->cpu_time) {
            printf("Process %s completed.\n", running->name);
            queue_pop_front(run_q);
        } else {
            running->state = W;
            queue_push_back(ready_q, queue_pop_front(run_q));
        }

        // 模拟阻塞队列中的进程唤醒
        if (block_q->head != NULL && ++count == counter) {
            PCB *blocked = &block_q->head->pcb;
            blocked->state = W;
            queue_push_back(ready_q, queue_pop_front(block_q));
            count = 0;
        }
    }
}

int main() {
    Queue ready_q, block_q, run_q;
    queue_init(&ready_q);
    queue_init(&block_q);
    queue_init(&run_q);

    PCB p1 = {"P1", 'W', 0, 0, 3, NULL};
    PCB p2 = {"P2", 'W', 0, 1, 6, NULL};
    PCB p3 = {"P3", 'B', 0, 2, 4, NULL};
    PCB p4 = {"P4", 'B', 0, 3, 5, NULL};
    PCB p5 = {"P5", 'B', 0, 4, 2, NULL};

    queue_push_back(&ready_q, p1);
    queue_push_back(&ready_q, p2);
    queue_push_back(&block_q, p3);
    queue_push_back(&block_q, p4);
    queue_push_back(&block_q, p5);

    dispatch(&ready_q, &block_q, &run_q);

    return 0;
}

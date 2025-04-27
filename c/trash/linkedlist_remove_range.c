#include <stdio.h>
#include <stdlib.h>

/*
假设以带头结点的单链表表示非递减有序表，
设计一算法删除表中所有值大于min且小于max
（假设min<max）同时释放结点空间。
*/

typedef struct __Node {
    struct __Node* next;
    int elem;
} Node;

typedef Node LinkedList;

void init(LinkedList* list) {
    list->next = NULL;
}

void add(LinkedList* list,int elem) {
    Node* node = list;
    while(node->next) {
        node = node->next;
    }
    node->next = (Node*)malloc(sizeof(Node));
    node->next->elem = elem;
    node->next->next = NULL;
}

void delete_range(LinkedList* list,int min,int max) {
    Node* node_l = NULL;
    Node* node_r = list->next;
    while(node_r->next) {
        if(node_r->elem >= max)
            return;

        if(node_r->elem > min && node_r->elem < max) {
            node_r = node_r->next;
            printf("remove (from first) %d\n",list->next->elem);
            free(list->next);
            list->next = node_r;
            node_l = list->next;
            continue;
        }
        node_l = list->next;

        if(node_r->elem > min && node_r->elem < max) {
            Node* p = node_r;
            node_r = node_r->next;
            printf("remove %d\n",p->elem);
            free(p);
            node_l->next = node_r;
        }
        else {
            node_l = node_l->next;
            node_r = node_r->next;
        }
    }
}

void show(LinkedList* list) {
    Node* node = list;
    while(node->next) {
        printf("node:{%d}\n",node->elem);
        node = node->next;
    }
}

int main(int, char**){
    LinkedList list;
    init(&list);
    for(int i = 0;i < 10;i++) {
        add(&list,i);
        add(&list,i);
        add(&list,i);
    }
    show(&list);
    delete_range(&list,-1,7);
    show(&list);
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 很烂，但是懒得改

int input[] = {13,15,22,8,34,19,21,29};

typedef struct __Node {
    struct __Node* next;
    int val;
} Node;

Node* arr[9];

int hash(int k) {
    int ret = (k % 7) + 1;
    printf("%d -> %d\n",k,ret);
    return ret;
}

void init() {
    //memset(arr,9,NULL);
    memset(arr,9,0);
}

void insert_linked(int val) {
    int hash_index = hash(val);
    if(!arr[hash_index]) {
        arr[hash_index] = (Node*)malloc(sizeof(Node));
        arr[hash_index]->next = NULL;
        arr[hash_index]->val = val;
    }
    else {
        int count = 1;
        Node* node = arr[hash_index];
        while(node->next) {
            node = node->next;
            ++count;
        }
        node->next = (Node*)malloc(sizeof(Node));
        node->next->next = NULL;
        node->next->val = val;
        printf("\tat %d (修正)\n",count);
    }
}

void test_linked() {
    for(int i = 0;i < sizeof(input) / sizeof(int);i++)
        insert_linked(input[i]);
}

int main(void) {
    test_linked();
    return 0;
}
#include <stdio.h>
#include <stdlib.h>

typedef struct __Block {
    int start_address;
    int length;
    struct __Block* prior;
    struct __Block* next;
} Block;

void show(Block* head) {
    printf("------------\n");
    Block* block = head;
    while(block) {
        printf("free block: start at %d, length %d\n",
            block->start_address,block->length);
        block = block->next;
    }
    printf("------------\n");
}

Block alloc(Block(*method)(Block*,int),Block* head,int size) {
    Block block = method(head,size);
    printf("allocated block at %d, len %d\n",block.start_address,block.length);
    show(head);
    return block;
}

Block first_fit(Block* head,int size) {
    Block* block = head;
    while(block) {
        if(block->length > size) {
            Block fit = {block->start_address,size,NULL,NULL};
            block->start_address = block->start_address + size;
            block->length = block->length - size;
            return fit;
        }
        else if(block->length == size) {
            Block fit = *block;
            fit.prior = fit.next = NULL;
            if(block->prior)
                block->prior->next = block->next;
            if(block->next)
                block->next->prior = block->prior;
            free(block);
            return fit;
        }
        block = block->next;
    }
    Block bad_block = {-1,-1,NULL,NULL};
    return bad_block;
}

Block best_fit(Block* head,int size) {
    
}

Block* free_block(Block* head,Block block) {
    int a,b,c,d;
    b = block.start_address;
    c = block.start_address + block.length;
    Block* ptr = head;
    Block* last = NULL;
    while(ptr) {
        d = ptr->start_address;
        if(ptr->prior) {
            a = ptr->prior->start_address + ptr->prior->length;
            if(a < b && c < d) {
                // 插入中间
                Block* new_block = (Block*)malloc(sizeof(Block));
                block.prior = ptr->prior;
                block.next = ptr;
                *new_block = block;
                ptr->prior->next = new_block;
                ptr->prior = new_block;
                printf("free address %d len %d by mid inserting\n",block.start_address,block.length);
                show(head);
                return head;
            }
            else if(a < b && c == d) {
                // 合并到右
                ptr->start_address = block.start_address;
                ptr->length += block.length;
                printf("free address %d len %d by merging to right\n",block.start_address,block.length);
                show(head);
                return head;
            }
            else if(a == b && c < d) {
                // 合并到左
                ptr->prior->length += block.length;
                printf("free address %d len %d by merging to left\n",block.start_address,block.length);
                show(head);
                return head;
            }
            else if(a == b && c == d) {
                // 左右合并
                ptr->prior->length += block.length + ptr->length;
                ptr->prior->next = ptr->next;
                ptr->next->prior = ptr->prior;
                free(ptr);
                printf("free address %d len %d by merging both sides\n",block.start_address,block.length);
                show(head);
                return head;
            }
        }
        else {
            if(c < d) {
                // 插入左侧
                Block* new_block = (Block*)malloc(sizeof(Block));
                block.prior = NULL;
                block.next = ptr;
                *new_block = block;
                ptr->prior = new_block;
                printf("free address %d len %d by creating a new head\n",block.start_address,block.length);
                show(new_block);
                return new_block;
            }
            else if(c == d) {
                // 合并到右
                ptr->start_address = block.start_address;
                ptr->length += block.length;
                printf("free address %d len %d by merging to head\n",block.start_address,block.length);
                show(head);
                return head;
            }
        }
        last = ptr;
        ptr = ptr->next;
    }
    if(last) {
        if(last->start_address + last->length == block.start_address) {
            // 合并到末尾
            last->length += block.length;
            printf("free address %d len %d by merging to last\n",block.start_address,block.length);
            show(head);
        }
        else {
            // 添加到末尾
            Block* new_block = (Block*)malloc(sizeof(Block));
            block.prior = last;
            block.next = NULL;
            *new_block = block;
            last->next = new_block;
            printf("free address %d len %d by insert to last\n",block.start_address,block.length);
            show(head);
        }
    }
    return head;
}

int init_block(Block* head,int length) {
    head->prior = NULL;
    head->next = NULL;
    head->start_address = 0;
    head->length = length;
}

void test_first_fit() {
    Block* head = (Block*)malloc(sizeof(Block));
    init_block(head,640);

    Block b1 = alloc(first_fit,head,130);
    Block b2 = alloc(first_fit,head,60);
    Block b3 = alloc(first_fit,head,100);
    head = free_block(head,b2);
    Block b4 = alloc(first_fit,head,200);
    head = free_block(head,b3);
    head = free_block(head,b1);
    Block b5 = alloc(first_fit,head,140);
    Block b6 = alloc(first_fit,head,60);
    Block b7 = alloc(first_fit,head,50);
    head = free_block(head,b6);

    show(head);
}

int main() {
    test_first_fit();
    return 0;
}
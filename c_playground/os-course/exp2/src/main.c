#include <stdio.h>
#include <stdlib.h>

typedef struct __Block {
    int start_address;
    int length;
    struct __Block* prior;
    struct __Block* next;
} Block;

Block alloc(Block(*method)(Block*,int),Block* head,int size) {
    return method(head,size);
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

/*
对由小到大的空闲区排列的空闲区，与所申请的内存大小相比，取两者差最小的给予分配，插入。
初始化最小空间和最佳位置，用ch来记录最小位置。
*/
Block best_fit(Block* head,int size) {
    // TODO
    Block bad_block = {-1,-1,NULL,NULL};
    return bad_block;
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
                return head;
            }
            else if(a < b && c == d) {
                // 合并到右
                ptr->start_address = block.start_address;
                ptr->length += block.length;
                return head;
            }
            else if(a == b && c < d) {
                // 合并到左
                ptr->prior->length += block.length;
                return head;
            }
            else if(a == b && c == d) {
                // 左右合并
                ptr->prior->length += block.length + ptr->length;
                ptr->prior->next = ptr->next;
                ptr->next->prior = ptr->prior;
                free(ptr);
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
                return new_block;
            }
            else if(c == d) {
                // 合并到右
                ptr->start_address = block.start_address;
                ptr->length += block.length;
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
        }
        else {
            // 添加到末尾
            Block* new_block = (Block*)malloc(sizeof(Block));
            block.prior = last;
            block.next = NULL;
            *new_block = block;
            last->next = new_block;
        }
    }
    return head;
}

void show(Block* head) {
    Block* block = head;
    while(block) {
        printf("free block: start at %d, length %d\n",
            block->start_address,block->length);
        block = block->next;
    }
}

int init_block(Block* head,int length) {
    head->prior = NULL;
    head->next = NULL;
    head->start_address = 0;
    head->length = length;
}

int main() {
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
    return 0;
}
/*
假设有一个循环链表的长度大于1，且表中既无头结点也无头指针。已知s为指向链表某个结点的指针，试编写算法在链表中删除指针s所指结点的前趋结点。
[提示]：设指针p指向s结点的前趋的前趋，则p与s有何关系？
*/

#include <stdio.h>
#include <stdlib.h>

typedef struct __CLinkListNode
{
    struct __CLinkListNode* next;
    struct __CLinkListNode* former;
    int val;
} CLinkListNode;

void clinklist_init(CLinkListNode* node)
{
    node->next = node;
    node->former = node;
    node->val = 0;
}

void clinklist_insert_front(CLinkListNode* node,int val)
{
    CLinkListNode* original_former = node->former;
    node->former = (CLinkListNode*)malloc(sizeof(CLinkListNode));
    node->former->former = original_former;
    node->former->next = node;
    node->former->val = val;
}

void clinklist_insert_back(CLinkListNode* node,int val)
{
    CLinkListNode* original_next = node->next;
    node->next = (CLinkListNode*)malloc(sizeof(CLinkListNode));
    node->next->next = original_next;
    node->next->former = node;
    node->next->val = val;
}

void clinklist_foreach(CLinkListNode* node,void(*callback)(CLinkListNode*))
{
    callback(node);
    CLinkListNode* next_node = node->next;
    while(next_node != node)
    {
        callback(next_node);
        next_node = next_node->next;
    }
}

void clinklist_remove(CLinkListNode* node,int val)
{
    CLinkListNode* moving_node = node;
    while(moving_node != node)
    {
        if(moving_node->val == val)
        {
            // 环形链表的next和former永远不应该为NULL
            // 只有一个节点时，其next和former都指向自身
            // TODO: 删除节点和维护缺口
        }
        moving_node = moving_node->next;
    };
}

int main(void)
{

}
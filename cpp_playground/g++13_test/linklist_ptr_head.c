/*
以单链表为存储结构实现以下基本操作：
(1)	在第i个结点前插入一个新结点。
(2)	查找值为x的某个元素。若成功，给出x在表中的位置；不成功给出提示信息。
(3)	删除第i个结点，若成功，给出提示信息并显示被删元素的值；不成功给出失败的提示信息。
*/

#include <stdio.h>
#include <stdlib.h>

typedef struct __LinkListNode
{
    struct __LinkListNode* next;
    int val;
} LinkListNode;

typedef LinkListNode* LinkListHead;

void linklist_init(LinkListHead* head)
{
    *head = NULL;
}

void linklist_insert(LinkListHead* head,int count,int val)
{
    if(count <= 0)
    {
        puts("position must be positive");
        puts("done nothing");
        return;
    }
    else if(!*(head))
    {
        puts("inserting into an empty linklist");
        printf("creating first node with var=%d\n",val);
        (*head) = (LinkListNode*)malloc(sizeof(LinkListNode));
        (*head)->next = NULL;
        (*head)->val = val;
    }
    else
    {
        LinkListNode* node = (*head);
        if(count == 1)
        {
            *head = (LinkListNode*)malloc(sizeof(LinkListNode));
            (*head)->val = val;
            (*head)->next = node;
            printf("val %d inserted to head\n",val);
        }
        else
        {
            for(int i = 0;i < count - 2;i++)
            {
                if(!node->next)
                {
                    printf("count %d out of range [1,%d)\n",count,i + 3);
                    puts("done nothing");
                    return;
                }
                else
                    node = node->next;
            }
            LinkListNode* new_node = (LinkListNode*)malloc(sizeof(LinkListNode));
            new_node->val = val;
            new_node->next = node->next;
            node->next = new_node;
            printf("val %d inserted\n",val);
        }
    }
}

int linklist_locate(LinkListHead head,int val)
{
    if(!head)
    {
        puts("empty linklist");
        puts("returning -1");
        return -1;
    }
    else
    {
        LinkListNode* node = head;
        for(int i = 0;node;i++)
        {
            if(node->val == val)
            {
                printf("located val %d at position %d\n",val,i + 1);
                return i;
            }
            node = node->next;
        }
        printf("can't locate val=%d from linklist\n",val);
        puts("returning -1");
        return -1;
    }
}

void linklist_remove(LinkListHead* head,int count)
{
    if(count <= 0)
    {
        puts("position must be positive");
        puts("done nothing");
        return;
    }
    else if(!(*head))
    {
        puts("empty linklist");
        puts("done nothing");
    }
    else
    {
        LinkListNode* node = (*head);
        if(count == 1)
        {
            printf("removing head node at 0x%p, val=%d, next=0x%p\n",*head,(*head)->val,(*head)->next);
            LinkListNode* second_node = (*head)->next;
            free(*head);
            (*head) = second_node;
        }
        else
        {
            for(int i = 0;i < count - 2;i++)
            {
                node = node->next;
                if(!node || !(node->next))
                {
                    printf("count %d out of range [1,%d)\n",count,i + 3);
                    puts("done nothing");
                    return;
                }
            }
            LinkListNode* removing_node = node->next;
            printf("removing node at 0x%p, val=%d, next=0x%p\n",removing_node,removing_node->val,removing_node->next);
            node->next = removing_node->next;
            free(removing_node);
        }        
    }
}

void linklist_show_all(LinkListHead head)
{
    if(!head)
        puts("empty");
    else
    {
        LinkListNode* node = head;
        while(node)
        {
            printf("%d ",node->val);
            node = node->next;
        }
        puts("");
    }
}

static LinkListHead head;

void menu_loop(void)
{
    int input,count,val;
    while(1)
    {
        puts("------ [ Menu ] ------");
        puts("1     => show all data");
        puts("2     => insert");
        puts("3     => locate");
        puts("4     => remove");
        puts("other => exit");
        printf("input=");
        scanf("%d",&input);

        switch (input)
        {
        case 1:
            puts("------ [ Show all ] ------");
            linklist_show_all(head);
            break;
        case 2:
            puts("------ [ Insert ] ------");
            printf("position = ");
            scanf("%d",&count);
            printf("val = ");
            scanf("%d",&val);
            linklist_insert(&head,count,val);
            break;
        case 3:
            puts("------ [ Locate ] ------");
            printf("val = ");
            scanf("%d",&val);
            linklist_locate(head,val);
            break;
        case 4:
            puts("------ [ Remove ] ------");
            printf("position = ");
            scanf("%d",&count);
            linklist_remove(&head,count);
            break;
        default:
            puts("------ [ Exit ] ------");
            return;
        }
    }
}

int main(void)
{
    linklist_init(&head);
    for(int i = 0;i < 5;i++)
        linklist_insert(&head,1,i);

    menu_loop();
}
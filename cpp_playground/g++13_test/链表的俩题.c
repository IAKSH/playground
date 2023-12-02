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

typedef LinkListNode LinkListHead;

void linklist_init(LinkListHead* head)
{
    head->next = NULL;
}

void linklist_insert(LinkListHead* head,int count,int val)
{
    if(count <= 0)
    {
        puts("position must be positive");
        puts("done nothing");
        return;
    }
    else if(!(head->next))
    {
        puts("inserting into an empty linklist");
        printf("creating first node with var=%d\n",val);
        head->next = (LinkListNode*)malloc(sizeof(LinkListNode));
        head->next->next = NULL;
        head->next->val = val;
    }
    else
    {
        LinkListNode* node = head->next;
        if(count == 1)
        {
            head->next = (LinkListNode*)malloc(sizeof(LinkListNode));
            head->next->val = val;
            head->next->next = node;
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

int linklist_locate(LinkListHead* head,int val)
{
    if(!(head->next))
    {
        puts("empty linklist");
        puts("returning -1");
        return -1;
    }
    else
    {
        LinkListNode* node = head->next;
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
    else if(!(head->next))
    {
        puts("empty linklist");
        puts("done nothing");
    }
    else
    {
        LinkListNode* node = head->next;
        if(count == 1)
        {
            printf("removing head node at 0x%p, val=%d, next=0x%p\n",head->next,head->next->val,head->next->next);
            LinkListNode* second_node = head->next->next;
            free(head->next);
            head->next = second_node;
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

void linklist_show_all(LinkListHead* head)
{
    if(!(head->next))
        puts("empty");
    else
    {
        LinkListNode* node = head->next;
        while(node)
        {
            printf("%d ",node->val);
            node = node->next;
        }
        puts("");
    }
}

void ordered_linklist_delete_same(LinkListHead* head)
{
    if(!head->next)
        puts("empty, exiting");
    else
    {
        LinkListNode* l_node = head->next;
        LinkListNode* r_node = head->next->next;
        while(r_node)
        {
            if(r_node->val == l_node->val)
            {
                LinkListNode* r_next_node = r_node->next;
                free(r_node);
                r_node = r_next_node;
                l_node->next = r_node;
                continue;
            }
            r_node = r_node->next;
            l_node = l_node->next;
        }
    }
}

void ordered_linklist_delete_range(LinkListHead* head,int mink,int maxk)
{
    if(!head->next)
        puts("empty, exiting");
    else
    {
        LinkListNode* l_node = head->next;
        LinkListNode* r_node = head->next->next;
        while(r_node)
        {
            if(l_node == head->next && (l_node->val > mink && l_node->val < maxk))
            {
                l_node = r_node;
                free(head->next);
                head->next = l_node;
                if(!(r_node->next))
                {
                    if(r_node->val > mink && r_node->val < maxk)
                    {
                        free(r_node);
                        head->next = NULL;
                    }
                    break;
                }
                else
                    r_node = r_node->next;
            }
            else if(r_node->val > mink && r_node->val < maxk)
            {
                LinkListNode* r_next_node = r_node->next;
                free(r_node);
                r_node = r_next_node;
                l_node->next = r_node;
            }
            else
            {
                r_node = r_node->next;
                l_node = l_node->next;
            }
        }
        if(l_node == head->next && (l_node->val > mink && l_node->val < maxk))
        {
            free(head->next);
            head->next = NULL;
        }
    }
}

static LinkListHead head;

int main(void)
{
    linklist_init(&head);
    //int _j = 0;
    //for(int i = 0;i < 10;i++)
    //{
    //    i += _j;
    //    linklist_insert(&head,i + 1,i + 1);
    //    if(i % 2 == 1)
    //    {
    //        int j;
    //        for(j = 0;j < i;++j)
    //            linklist_insert(&head,i + 1,i);
    //        _j = j;
    //    }
    //}

    linklist_insert(&head,1,1);

    linklist_show_all(&head);
    //ordered_linklist_delete_same(&head);
    ordered_linklist_delete_range(&head,-10,100);
    linklist_show_all(&head);
}
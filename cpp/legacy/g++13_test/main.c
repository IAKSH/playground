#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    int max_len;
    int len;
    int arr[10];
} SeqList;

void seqlist_insert(SeqList* list,int index,int val)
{
    if (index > list->max_len)
        fprintf(stderr,"index out of range, done nothing\n");
    else if (list->len >= list->max_len)
        fprintf(stderr,"SeqList full, done nothing\n");
    else
    {
        for(int i = list->len++;i > index;i--)
            list->arr[i] = list->arr[i - 1];
        list->arr[index] = val;
    }
}

int seqlist_get_index(SeqList* list,int val)
{
    for(int i = 0;i < list->len;i++)
    {
        if(list->arr[i] == val)
            return i;
    }
    fprintf(stderr,"value %d not found, returning -1\n",val);
    return -1;
}

void seqlist_delete(SeqList* list,int index)
{
    if(index >= list->len)
        fprintf(stderr,"index out of range, done nothing\n");
    else
    {
        for(int i = index;i < list->len - 1;i++)
            list->arr[i] = list->arr[i + 1];
        --list->len;
    }
}

void seqlist_init(SeqList* list)
{
    list->max_len = 10;
    list->len = 0;
    memset(list->arr,0,sizeof(int) * list->len);
}

void seqlist_show(SeqList* list)
{
    for(size_t i = 0;i < list->len;i++)
        printf("%d ",list->arr[i]);
    puts("\n");
}

void test_1(void)
{
    printf("--------- TEST 1 ---------\n");

    SeqList list;
    seqlist_init(&list);

    for(int i = 0;i < 11;i++)
        seqlist_insert(&list,0,i);

    seqlist_show(&list);

    printf("trying to found index of %d: %d\n",4,seqlist_get_index(&list,4));
    printf("trying to found index of %d: %d\n",-1,seqlist_get_index(&list,-1));

    printf("delete at 0\n");
    seqlist_delete(&list,0);
    seqlist_show(&list);

    printf("delete at 2\n");
    seqlist_delete(&list,2);
    seqlist_show(&list);

    printf("delete at the last\n");
    seqlist_delete(&list,list.len - 1);
    seqlist_show(&list);
}

void test_2(void)
{
    printf("--------- TEST 2 ---------\n");

    SeqList list;
    seqlist_init(&list);

    for(int i = 0;i < 11;i++)
        seqlist_insert(&list,0,(i + 1) * 2);

    seqlist_show(&list);

    printf("trying to found index of %d: %d\n",4,seqlist_get_index(&list,4));
    printf("trying to found index of %d: %d\n",-1,seqlist_get_index(&list,-1));

    printf("delete at 0\n");
    seqlist_delete(&list,0);
    seqlist_show(&list);

    printf("delete at 2\n");
    seqlist_delete(&list,2);
    seqlist_show(&list);

    printf("delete at the last\n");
    seqlist_delete(&list,list.len - 1);
    seqlist_show(&list);
}

int main(void)
{
    test_1();
    test_2();
}
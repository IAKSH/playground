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
    if (index < 0)
        fprintf(stderr,"index must be a natural number, done nothing\n");
    else if (index > list->len)
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
    if (index < 0)
        fprintf(stderr,"index must be a natural number, done nothing\n");
    else if(index >= list->len)
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
    memset(list->arr,0,sizeof(int) * list->max_len);
}

void seqlist_show(SeqList* list)
{
    for(size_t i = 0;i < list->len;i++)
        printf("%d ",list->arr[i]);
    puts("\n");
}

void menu_loop()
{
    SeqList list;
    seqlist_init(&list);

    for(int i = 0;i < 8;i++)
        seqlist_insert(&list,0,i);

    int input_flag;
    while(1)
    {
        puts("--------- MENU ---------");
        puts("input:");
        puts("0 -> show");
        puts("1 -> insert");
        puts("2 -> delete");
        puts("3 -> get");
        puts("others -> exit");

        scanf("%d",&input_flag);

        switch(input_flag)
        {
        case 0:
            {
                puts("--------- SHOW ---------");
                seqlist_show(&list);
                break;
            }
        
        case 1:
            {
                puts("--------- INSERT ---------");
                int index,val;
                printf("index: ");scanf("%d",&index);
                printf("value: ");scanf("%d",&val);
                seqlist_insert(&list,index,val);
                break;
            }

        case 2:
            {
                puts("--------- DELETE ---------");
                int index;
                printf("index: ");scanf("%d",&index);
                seqlist_delete(&list,index);
                break;
            }

        case 3:
            {
                puts("--------- GET ---------");
                int val;
                printf("value: ");scanf("%d",&val);
                printf("result: %d\n",seqlist_get_index(&list,val));
                break;
            }

        default:
            return;
        }
    }
}

int main(void)
{
    menu_loop();
}
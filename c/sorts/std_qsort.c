#include <stdio.h>
#include <stdlib.h>

int compare(const void* n,const void* m) {
    return *((const int*)n) - *((const int*)m);
}

int main(void) {
    int arr[10];
    for(int i = 0;i < 10;i++)
        arr[i] = rand() % 50;

    for(int i = 0;i < 10;i++)
        printf("%d ",arr[i]);
    puts("");

    qsort(arr,10,sizeof(int),compare);

    for(int i = 0;i < 10;i++)
        printf("%d ",arr[i]);
    puts("");

    return 0;
}
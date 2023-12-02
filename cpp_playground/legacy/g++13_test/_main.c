// 删除顺序表中x到y之间的元素，要求空间复杂度为O（1），时间复杂度O（n）。

#include <stdio.h>
#include <stdlib.h>

void delete_range(int* arr,size_t len,size_t* count,int m,int n) {
    if (*count > len) {
        fprintf(stderr,"count > len");
        return;
    }

    int i,j;
    for(i = m,j = n + 1;j < *count;j++) {
        arr[i++] = arr[j];
    }

    *count -= j - i;
}

int main() {
    int arr[8] = {0,1,2,3,4,5,6,7};
    size_t count = 8;
    delete_range(arr,8,&count,2,5);

    for(int i = 0;i < count;i++)
        printf("%d, ",arr[i]);

    printf("\b\b \n");
    return 0;
}
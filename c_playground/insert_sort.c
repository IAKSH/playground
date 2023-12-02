#include <stdio.h>

void insert_sort(int* arr,size_t len) {
    for(size_t i = 1; i < len; i++) {           // 对于arr[i]（[1,end)），i实际上就是有序区域的有边界，i之前都有序。
        for(size_t j = 0; j < i; j++) {         // 在arr的[0,i)范围（有序区域）内寻找第一个大于arr[i]的值
            if (arr[j] > arr[i]) {              // 找到有序区域内第一个大于arr[i]的值arr[j]
                int val = arr[i];
                for(int k = i; k != j; k--)     // 将有序区的[j,i)部分整体右移，占据arr[i]为有序区域，同时空出arr[j]，然后插入
                    arr[k] = arr[k - 1];
                arr[j] = val;
                break;
            }
        }
    }
}

int main(void) {
    int arr[] = {12,24,534,2,123,5,3,7,2,-123,0,23};

    printf("before: {");
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        printf("%d,",arr[i]);
    puts("\b}");

    insert_sort(arr,sizeof(arr)/sizeof(int));

    printf("after: {");
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        printf("%d,",arr[i]);
    puts("\b}");
    
    return 0;
}
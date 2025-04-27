#include <stdio.h>
#include <limits.h>

void select_sort(int* arr,size_t len) {
    for(int i = 0;i < len;i++) {
        int min = INT_MAX;
        int min_index = i;
        for(int j = i;j < len;j++) {
            if(arr[j] < min) {
                min = arr[j];
                min_index = j;
            }
        }
        if(min_index != i) {                // 小小的优化，避免不必要的交换
            arr[min_index] = arr[i];
            arr[i] = min;
        }
    }
}

int main(void) {
    int arr[] = {12,24,534,2,123,5,3,7,2,-123,0,23};

    printf("before: {");
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        printf("%d,",arr[i]);
    puts("\b}");

    select_sort(arr,sizeof(arr)/sizeof(int));

    printf("after: {");
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        printf("%d,",arr[i]);
    puts("\b}");
    
    return 0;
}
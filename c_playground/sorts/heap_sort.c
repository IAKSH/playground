#include <stdio.h>

void swap(int* a, int* b) {
    int temp = *b;
    *b = *a;
    *a = temp;
}

void max_heapify(int arr[], int start, int end) {
    int dad = start;
    int son = dad * 2 + 1;
    while (son <= end) {                                    //若子节点指标在范围内才做比较
        if (son + 1 <= end && arr[son] < arr[son + 1])      //先比较两个子节点大小，选择最大的
            son++;
        if (arr[dad] > arr[son])                            //如果父节点大于子节点代表调整完毕，直接跳出函数
            return;
        else {                                              //否则交换父子内容再继续子节点和孙节点比较
            swap(&arr[dad], &arr[son]);
            dad = son;
            son = dad * 2 + 1;
        }
    }
}

void heap_sort(int* arr,size_t len) {
    int i;
    for (i = len / 2 - 1; i >= 0; i--)                                      // 易得最后一个非叶节点的下标为 n/2-1，n为表长
        max_heapify(arr, i, len - 1);                                       // 对每个非叶节点进行一次下滤，建立大根堆
    for (i = len - 1; i > 0; i--) {
        swap(&arr[0], &arr[i]);                                             // 交换堆顶与数组尾元素
        max_heapify(arr, 0, i - 1);                                         // 缩小堆范围并将堆顶重新下滤以维护大根堆
    }
}

int main(void) {
    int arr[] = {12,24,534,2,123,5,3,7,2,-123,0,23};
    printf("before:\t{");
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        printf("%d,",arr[i]);
    puts("\b}");

    heap_sort(arr,sizeof(arr)/sizeof(int));

    printf("after:\t{");
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        printf("%d,",arr[i]);
    puts("\b}");
    
    return 0;
}
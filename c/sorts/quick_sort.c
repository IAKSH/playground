#include <stdio.h>

void quick_sort(int* arr,size_t len) {
    if(len <= 1)
        return;

    typedef struct {                    // 包装一个Range，然后以FIFO形式储存它
        size_t l,r;
    } Range;
    
    int ranges_i,ranges_j;
    Range ranges[len];                  // 拿数组当一个循环队列，就像是二叉树的广度优先遍历
    ranges_i = 0;                       // 队列头部
    ranges_j = 0;                       // 队列尾部

    // add the first range
    ranges[ranges_j].l = 0;
    ranges[ranges_j++].r = len - 1;

    // process all ranges (and maybe add more)
    while((ranges_i % len < ranges_j % len)) {
        if(ranges[ranges_i % len].l < ranges[ranges_i % len].r) {
            int pivot = arr[ranges[ranges_i % len].l];
            size_t i = ranges[ranges_i % len].l;
            size_t j = ranges[ranges_i % len].r;
            while(i < j) {
                while (i < j && arr[j] >= pivot) --j;
                arr[i] = arr[j];
                while (i < j && arr[i] <= pivot) ++i;
                arr[j] = arr[i];
            }
            arr[i] = pivot;
            // add left range
            ranges[ranges_j % len].l = ranges[ranges_i % len].l;
            ranges[ranges_j++ % len].r = i - 1;
            // add right range
            ranges[ranges_j % len].l = i + 1;
            ranges[ranges_j++ % len].r = ranges[ranges_i % len].r;
        }
        ranges_i++;
    }
}

int main(void) {
    int arr[] = {12,24,534,2,123,5,3,7,2,-123,0,23};
    printf("before: {");
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        printf("%d,",arr[i]);
    puts("\b}");

    quick_sort(arr,sizeof(arr)/sizeof(int));

    printf("after: {");
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        printf("%d,",arr[i]);
    puts("\b}");
    
    return 0;
}
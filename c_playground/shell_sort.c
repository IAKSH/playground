#include <stdio.h>

// 希尔排序实际上是分组的插入排序，通过分组来减少插入排序的比较和插入次数
// 虽然并不快，但是希尔排序有重大的历史意义
// 因为是人类历史上第一个突破O(N^2)时间复杂度的排序算法
// 可以用递归写，不过会很慢
void shell_sort(int* arr,size_t len) {
    // 大概逻辑如下：
    // 1. 分组（每组的成员都是散布的）
    // 2. 各组内进行插入排序
    // 3. 调整组间隔，循环
    // 具体怎么调整组间隔：在样本较少时，可以直接/=2
    // 但其他情况下，希尔本人的意见是取2^x-1序列 (1 3 7 15...)，因为这样可以在gap=1之前就进行奇数位置和偶数位置的比较，则在进行最后的全体插入排序时，数组能够更加有序
    //                                                                                                    （注：插入排序在nearly sorted时的时间复杂度接近O(n)）
    // 但是后来的更多文章推荐使用3x+1序列 (1 4 13 40 121...)

    // 初始组间隔取 len / 2，然后每次/=2，直到组间隔为1（整个数组）时进行最后一次插入排序
    for(int gap = len / 2;gap > 0;gap /= 2) {
        // 每次先取i于一个比较中间的位置（尽量让前后都有组员），然后按照gap寻找组员，并逐个施以插入排序
        for (int i = gap; i < len; i++) {
            int temp = arr[i];
            int j;
            for (j = i; j >= gap && arr[j - gap] > temp; j -= gap)
                arr[j] = arr[j - gap];
            arr[j] = temp;
        }
    }
}

int main(void) {
    int arr[] = {12,24,534,2,123,5,3,7,2,-123,0,23};

    printf("before: {");
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        printf("%d,",arr[i]);
    puts("\b}");

    shell_sort(arr,sizeof(arr)/sizeof(int));

    printf("after: {");
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        printf("%d,",arr[i]);
    puts("\b}");
    
    return 0;
}
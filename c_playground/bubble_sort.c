#include <stdio.h>

// 带优化的冒泡，基本有序(nearly sorted)时的时间复杂度接近O(n)
void bubble_sort(int* arr,size_t len) {
    for(size_t i = 0; i < len; i++) {               // 实际上就是依次把最大元素逐层推到末尾，在末尾形成一个有序区域
        int swaped = 0;                             // 上面的这个i并不是对每个arr[i]进行操作，i实际上是arr末端的有序区域的长度
        for(size_t j = 0; j < len - i - 1; j++) {   // 由于末尾是有序区，所以只需要在其之前的无序区域中将最大值“冒泡”，浮到有序区域的尾部
            if(arr[j + 1] < arr[j]) {               // 无序区域的最大数的"气泡"就是通过交换逐渐“上浮”的
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
                swaped = 1;
            }
        }
        if(!swaped)                                 // 优化在此，某次对无序区的遍历全程未发生交换，则确认已经有序，提前结束
            break;
    }
}

int main(void) {
    int arr[] = {12,24,534,2,123,5,3,7,2,-123,0,23};
    printf("before: {");
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        printf("%d,",arr[i]);
    puts("\b}");

    bubble_sort(arr,sizeof(arr)/sizeof(int));

    printf("after: {");
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        printf("%d,",arr[i]);
    puts("\b}");
    
    return 0;
}
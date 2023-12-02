#include <stdio.h>

//#define PRINT_RECURSION_DEPTH

#ifdef PRINT_RECURSION_DEPTH
static int depth = 0;
#endif

void quick_sort_recursion(int* arr,size_t l,size_t r) {
#ifdef PRINT_RECURSION_DEPTH
    ++depth;
#endif
    if(l < r) {                                             // 当l,r迭代到同一点，即已经分治到最小区域时，排序完成
        // one pass
        int pivot = arr[l];                                 // 单次排序选定当前区域的第一个元素为轴
        size_t i = l;
        size_t j = r;
        while(i < j) {                                      // 一直迭代到i，j相撞
            while (i < j && arr[j] >= pivot) --j;           // 跳过不需要移动的值
            arr[i] = arr[j];                                // 由于其中一个是取出轴值后留下的空位，所以只需要单向复制
            while (i < j && arr[i] <= pivot) ++i;           // 跳过不需要移动的值
            arr[j] = arr[i];                                // 同上
        }
        arr[i] = pivot;                                     // i，j相撞之处就是中轴值应该在的地方，此时其左侧都比他小，右侧都比他大
        // recursion left
        quick_sort_recursion(arr,l,i - 1);                  // 对左侧区域递归
        // recursion right
        quick_sort_recursion(arr,i + 1,r);                  // 对右侧区域递归
    }
}

void quick_sort(int* arr,size_t len) {
    quick_sort_recursion(arr,0,len - 1);
}

int main(void) {
    int arr[] = {12,24,534,2,123,5,3,7,2,-123,0,23};
    printf("before: {");
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        printf("%d,",arr[i]);
    puts("\b}");

    quick_sort(arr,sizeof(arr)/sizeof(int));
#ifdef PRINT_RECURSION_DEPTH
    printf("递归深度：%d\n",depth);
#endif

    printf("after: {");
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        printf("%d,",arr[i]);
    puts("\b}");
    
    return 0;
}
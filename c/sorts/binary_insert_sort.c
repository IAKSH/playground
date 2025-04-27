#include <stdio.h>

void binary_insert_sort(int* arr,size_t len) {
    int l,r,m;
    for(size_t i = 1; i < len; i++) {
        l = 0;
        r = i - 1;
        while(l <= r) {                 // l == r时依然执行，即可根据arr[i]与arr[l]（即arr[m]）的大小确定是插在其左还是右侧
            m = (l + r) / 2;
            if(arr[i] < arr[m])
                r = m - 1;
            else
                l = m + 1;
        }
        int val = arr[i];
        for(int j = i; j != l; j--)     // 将有序区的[l,i)部分整体右移，占据arr[i]为有序区域，同时空出arr[j]，然后插入
            arr[j] = arr[j - 1];
        arr[l] = val;
    }
}

int main(void) {
    int arr[] = {12,24,534,2,123,5,3,7,2,-123,0,23};

    printf("before: {");
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        printf("%d,",arr[i]);
    puts("\b}");

    binary_insert_sort(arr,sizeof(arr)/sizeof(int));

    printf("after: {");
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        printf("%d,",arr[i]);
    puts("\b}");
    
    return 0;
}
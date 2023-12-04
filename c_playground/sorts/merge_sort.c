
#include <stdio.h>

#define MIN(x,y) ((x) < (y) ? (x) : (y))

void merge_sort(int* arr,size_t len) {
    int buffer[len];
    int* a = arr;
    int* b = buffer;
    int seg, start;
    for (seg = 1; seg < len; seg += seg) {
        for (start = 0; start < len; start += seg + seg) {
            int low = start, mid = MIN(start + seg, len), high = MIN(start + seg + seg, len);
            int k = low;
            int start1 = low, end1 = mid;
            int start2 = mid, end2 = high;
            while (start1 < end1 && start2 < end2)
                b[k++] = a[start1] < a[start2] ? a[start1++] : a[start2++];
            while (start1 < end1)
                b[k++] = a[start1++];
            while (start2 < end2)
                b[k++] = a[start2++];
        }
        int* temp = a;
        a = b;
        b = temp;
    }
    if (a != arr) {
        int i;
        for (i = 0; i < len; i++)
            b[i] = a[i];
        b = a;
    }
}

int main(void) {
    int arr[] = {12,24,534,2,123,5,3,7,2,-123,0,23};
    printf("before:\t{");
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        printf("%d,",arr[i]);
    puts("\b}");

    merge_sort(arr,sizeof(arr)/sizeof(int));

    printf("after:\t{");
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        printf("%d,",arr[i]);
    puts("\b}");
    
    return 0;
}
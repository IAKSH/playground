#include <stdio.h>

void merge_sort_recursion(int* arr,int* buffer,size_t l,size_t r) {
    if(r - l <= 1) return;
    // recursion
    merge_sort_recursion(arr,buffer,l,(l + r) / 2);
    merge_sort_recursion(arr,buffer,(l + r) / 2,r);
    // sort to buffer
    int i = l;
    int j = (l + r) / 2;
    int k = l;
    while(k != r) {
        if(i >= (l + r) / 2) {
            for(;j < r;j++)
                buffer[k++] = arr[j];
            break;
        }
        else if(j >= r) {
            for(;i < (l + r) / 2;i++)
                buffer[k++] = arr[i];
            break;
        }
        else if(arr[j] < arr[i]) {
            buffer[k++] = arr[j++];
        }
        else {
            buffer[k++] = arr[i++];
        }
    }
    // apply buffer
    for(i = l;i < r;i++)
        arr[i] = buffer[i];
}

void merge_sort(int* arr,size_t len) {
    int buffer[len];
    merge_sort_recursion(arr,buffer,0,len);
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
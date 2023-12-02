#include <stdio.h>
#include <limits.h>
#include <string.h>

/*
void count_sort(int* arr,size_t len) {
    int max = INT_MIN;
    int min = INT_MAX;
    for(int i = 0;i < len;i++) {
        if(arr[i] > max)
            max = arr[i];
        if(arr[i] < min)
            min = arr[i];
    }

    int delta = 0 - min;
    int buffer[max - min];
    memset(buffer,0,sizeof(int) * (max - min));
    for(int i = 0;i < len;i++)
        buffer[arr[i] + delta]++;

    int i = 0;
    for(int j = 0;j < max;j++) {
        while(buffer[j]-- > 0)
            arr[i++] = j - delta;
    }
}
*/

void count_sort(int* arr,size_t len) {
    int max = INT_MIN;
    int min = INT_MAX;
    for(int i = 0;i < len;i++) {
        if(arr[i] > max)
            max = arr[i];
        if(arr[i] < min)
            min = arr[i];
    }

    int delta = 0 - min;
    int buffer[max - min + 1];
    memset(buffer,0,sizeof(int) * (max - min + 1));
    for(int i = 0;i < len;i++)
        buffer[arr[i] + delta]++;

    int i = 0;
    for(int j = 0;j <= max - min;j++) {
        while(buffer[j]-- > 0)
            arr[i++] = j + min;
    }
}

int main(void) {
    int arr[] = {12,24,534,2,123,5,3,7,2,-123,0,23};
    printf("before: {");
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        printf("%d,",arr[i]);
    puts("\b}");

    count_sort(arr,sizeof(arr)/sizeof(int));

    printf("after: {");
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        printf("%d,",arr[i]);
    puts("\b}");
    
    return 0;
}
#include <stdio.h>
#include <limits.h>
#include <string.h>

void radix_sort(int* arr, size_t len) {
    if (len <= 0)
        return;
    // find the minimum value in the array
    int min_val = arr[0];
    for (size_t i = 1; i < len; i++) {
        if (arr[i] < min_val) {
            min_val = arr[i];
        }
    }
    // if the minimum value is less than 0, offset all numbers by its absolute value
    // this is far more better than using a larger array and offset all elems' index
    int offset = 0;
    if (min_val < 0) {
        offset = -min_val;
        for (size_t i = 0; i < len; i++) {
            arr[i] += offset;
        }
    }
    // the rest of the radix sort code remains the same...
    int max_val = arr[0];
    for (size_t i = 1; i < len; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    // get the maximum oridinal, which equals to the count of sort
    int max_ordinal = 0;
    while (max_val != 0) {
        max_val /= 10;
        max_ordinal++;
    }
    // create radix buffer
    // we use a one-dimensional array to linearly store the elements obtained by a round of radix sort
    int output[len];
    int count[10] = {0};
    // real sort
    int exp = 1;
    for (int i = 0; i < max_ordinal; i++) {
        memset(count, 0, sizeof(count));
        for (size_t j = 0; j < len; j++) {
            int index = (arr[j] / exp) % 10;
            count[index]++;
        }
        for (int j = 1; j < 10; j++) {
            count[j] += count[j - 1];
        }
        for (int j = len - 1; j >= 0; j--) {
            int index = (arr[j] / exp) % 10;
            output[count[index] - 1] = arr[j];
            count[index]--;
        }
        // because we are storing elems using linearly array, so we can just simply copy all of them to original input arr
        // in current radix, they are alrealy sorted
        memcpy(arr, output, len * sizeof(int));
        exp *= 10;
    }
    // after the sorting is done, subtract the offset
    if (offset > 0) {
        for (size_t i = 0; i < len; i++) {
            arr[i] -= offset;
        }
    }
}

int main(void) {
    int arr[] = {12,24,534,2,123,5,3,7,2,-123,0,23};
    printf("before:\t{");
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        printf("%d,",arr[i]);
    puts("\b}");

    radix_sort(arr,sizeof(arr)/sizeof(int));

    printf("after:\t{");
    for(int i = 0;i < sizeof(arr)/sizeof(int);i++)
        printf("%d,",arr[i]);
    puts("\b}");
    
    return 0;
}
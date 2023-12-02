#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

inline static float get_max_i8(int8_t* arr,size_t len)
{
    int8_t max = INT8_MIN;
    for(size_t i = 0;i < len;i++)
        max = (arr[i] > max) ? arr[i] : max;
    return max;
}

inline static float get_max_u8(uint8_t* arr,size_t len)
{
    uint8_t max = 0;
    for(size_t i = 0;i < len;i++)
        max = (arr[i] > max) ? arr[i] : max;
    return max;
}

inline static float get_max_i16(int16_t* arr,size_t len)
{
    int16_t max = INT16_MIN;
    for(size_t i = 0;i < len;i++)
        max = (arr[i] > max) ? arr[i] : max;
    return max;
}

inline static float get_max_u16(uint16_t* arr,size_t len)
{
    uint16_t max = 0;
    for(size_t i = 0;i < len;i++)
        max = (arr[i] > max) ? arr[i] : max;
    return max;
}

inline static float get_max_i32(int32_t* arr,size_t len)
{
    int32_t max = INT32_MIN;
    for(size_t i = 0;i < len;i++)
        max = (arr[i] > max) ? arr[i] : max;
    return max;
}

inline static float get_max_u32(uint32_t* arr,size_t len)
{
    uint32_t max = 0;
    for(size_t i = 0;i < len;i++)
        max = (arr[i] > max) ? arr[i] : max;
    return max;
}

inline static float get_max_f(float* arr,size_t len)
{
    float max = INT_MIN;
    for(size_t i = 0;i < len;i++)
        max = (arr[i] > max) ? arr[i] : max;
    return max;
}

inline static float get_max_d(double* arr,size_t len)
{
    double max = INT_MIN;
    for(size_t i = 0;i < len;i++)
        max = (arr[i] > max) ? arr[i] : max;
    return max;
}

inline static void sorts_counting_heap_i32(int32_t* arr,size_t len,bool upper)
{
    int32_t max = get_max_i32(arr,len);
    int* counting_array = (int*)calloc(max,sizeof(int));

    memset(counting_array,0,max);
    for(size_t i = 0;i < len;i++)
        ++counting_array[arr[i]];

    size_t index = 0;
    for(size_t i = 0;(i < max) && (index < len);i++)
    {
        while(counting_array[i]--)
            arr[index++] = i;
    }

    if(!upper)
    {
        for(size_t i = 0;i < len / 2;i++)
        {
            int32_t buffer = arr[i];
            arr[i] = arr[len - 1 - i];
            arr[len - 1 - i] = buffer;
        }
    }

    free(counting_array);
}
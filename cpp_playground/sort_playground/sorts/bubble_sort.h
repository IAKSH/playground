#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

inline static void sorts_bubble_f(float* arr,size_t len,bool upper)
{
    for(size_t i = 0;i < len - 1;i++)
    {
        for(size_t j = 0;j < len - 1 - i;j++)
        {
            if((!upper)?(arr[j + 1] > arr[j]):(arr[j + 1] < arr[j]))
            {
                float buffer = arr[j + 1];
                arr[j + 1] = arr[j];
                arr[j] = buffer;
            }
        }
    }
}

inline static void sorts_bubble_d(double* arr,size_t len,bool upper)
{
    for(size_t i = 0;i < len - 1;i++)
    {
        for(size_t j = 0;j < len - 1 - i;j++)
        {
            if((!upper)?(arr[j + 1] > arr[j]):(arr[j + 1] < arr[j]))
            {
                double buffer = arr[j + 1];
                arr[j + 1] = arr[j];
                arr[j] = buffer;
            }
        }
    }
}

inline static void sorts_bubble_i32(int32_t* arr,size_t len,bool upper)
{
    for(size_t i = 0;i < len - 1;i++)
    {
        for(size_t j = 0;j < len - 1 - i;j++)
        {
            if((!upper)?(arr[j + 1] > arr[j]):(arr[j + 1] < arr[j]))
            {
                int32_t buffer = arr[j + 1];
                arr[j + 1] = arr[j];
                arr[j] = buffer;
            }
        }
    }
}

inline static void sorts_bubble_i16(int16_t* arr,size_t len,bool upper)
{
    for(size_t i = 0;i < len - 1;i++)
    {
        for(size_t j = 0;j < len - 1 - i;j++)
        {
            if((!upper)?(arr[j + 1] > arr[j]):(arr[j + 1] < arr[j]))
            {
                int16_t buffer = arr[j + 1];
                arr[j + 1] = arr[j];
                arr[j] = buffer;
            }
        }
    }
}

inline static void sorts_bubble_i8(int8_t* arr,size_t len,bool upper)
{
    for(size_t i = 0;i < len - 1;i++)
    {
        for(size_t j = 0;j < len - 1 - i;j++)
        {
            if((!upper)?(arr[j + 1] > arr[j]):(arr[j + 1] < arr[j]))
            {
                int8_t buffer = arr[j + 1];
                arr[j + 1] = arr[j];
                arr[j] = buffer;
            }
        }
    }
}

inline static void sorts_bubble_u32(uint32_t* arr,size_t len,bool upper)
{
    for(size_t i = 0;i < len - 1;i++)
    {
        for(size_t j = 0;j < len - 1 - i;j++)
        {
            if((!upper)?(arr[j + 1] > arr[j]):(arr[j + 1] < arr[j]))
            {
                uint32_t buffer = arr[j + 1];
                arr[j + 1] = arr[j];
                arr[j] = buffer;
            }
        }
    }
}

inline static void sorts_bubble_u16(uint16_t* arr,size_t len,bool upper)
{
    for(size_t i = 0;i < len - 1;i++)
    {
        for(size_t j = 0;j < len - 1 - i;j++)
        {
            if((!upper)?(arr[j + 1] > arr[j]):(arr[j + 1] < arr[j]))
            {
                uint16_t buffer = arr[j + 1];
                arr[j + 1] = arr[j];
                arr[j] = buffer;
            }
        }
    }
}

inline static void sorts_bubble_u8(uint8_t* arr,size_t len,bool upper)
{
    for(size_t i = 0;i < len - 1;i++)
    {
        for(size_t j = 0;j < len - 1 - i;j++)
        {
            if((!upper)?(arr[j + 1] > arr[j]):(arr[j + 1] < arr[j]))
            {
                uint8_t buffer = arr[j + 1];
                arr[j + 1] = arr[j];
                arr[j] = buffer;
            }
        }
    }
}

inline static void sorts_bubble(void* arr,size_t len,bool upper,
    bool(compare_callback)(void* m,void* n),
    void(swap_callback)(void* bigger,void* lower))
{
    for(size_t i = 0;i < len - 1;i++)
    {
        for(size_t j = 0;j < len - 1 - i;j++)
        {
            if(upper)
            {
                if(compare_callback(arr + j + 1,arr + j))
                    swap_callback(arr + j + 1,arr + j);
            }
            else
            {
                if(compare_callback(arr + j,arr + j + 1))
                    swap_callback(arr + j,arr + j + 1);
            }
        }
    }
}
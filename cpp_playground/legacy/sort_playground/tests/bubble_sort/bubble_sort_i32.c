#include <bubble_sort.h>
#include <stdio.h>
#include <stdlib.h>

#define TEST_TYPE int32_t
#define SORT_FUNC sorts_bubble_i32

static TEST_TYPE arr[10] = {3,2,1,6,5,4,9,8,7,0};

bool check_upper()
{
    for(int i = 0;i < 10;i++)
    {
        if(arr[i] !=  i)
            return false;
    }

    return true;
}

bool check_lower()
{
    for(int i = 0;i < 10;i++)
    {
        if(arr[i] !=  9 - i)
            return false;
    }

    return true;
}

int main(void)
{
    SORT_FUNC(arr,10,true);
    if(!check_upper())
        return EXIT_FAILURE;

    SORT_FUNC(arr,10,false);
    if(!check_lower())
        return EXIT_FAILURE;
}
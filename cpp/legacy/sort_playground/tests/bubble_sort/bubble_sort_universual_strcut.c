#include <bubble_sort.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct 
{
    unsigned int id;
    float x,y,z;
}
Object;

#define TEST_TYPE Object
#define SORT_FUNC sorts_bubble

static TEST_TYPE arr[10] = {{3},{2},{1},{6},{5},{4},{9},{8},{7},{0}};

bool check_upper()
{
    for(int i = 0;i < 10;i++)
    {
        if(arr[i].id !=  i)
            return false;
    }

    return true;
}

bool check_lower()
{
    for(int i = 0;i < 10;i++)
    {
        if(arr[i].id !=  9 - i)
            return false;
    }

    return true;
}

bool compare(void* m,void* n)
{
    return (unsigned int*)m > (unsigned int*)n;

}

void swap(void* bigger,void* lower)
{
    TEST_TYPE buffer = *(TEST_TYPE*)lower;
    *(TEST_TYPE*)lower = *(TEST_TYPE*)bigger;
    *(TEST_TYPE*)bigger = buffer;
}

int main(void)
{
    SORT_FUNC(arr,10,true,compare,swap);
    if(!check_upper())
        return EXIT_FAILURE;

    SORT_FUNC(arr,10,false,compare,swap);
    if(!check_lower())
        return EXIT_FAILURE;
}
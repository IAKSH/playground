#include <stdio.h>
#include <limits.h>

void sort(int* arr, size_t count)
{
    int i, j, buffer, minIndex, min;
    for (i = 0; i < count - 1; i++)
    {
        min = INT_MAX;
        for (j = i; j < count; j++)
        {
            if (arr[j] < min)
            {
                min = arr[j];
                minIndex = j;
            }
        }
        //swap
        buffer = arr[i];
        arr[i] = min;
        arr[minIndex] = buffer;
    }
}

int main(void)
{
	int arr[10];
	for (int i = 0; i < 10; i++)
		scanf("%d", arr + i);

	sort(arr, 10);
	for (int i = 9; i >= 0; i--)
		printf("%d\t", arr[i]);

	puts("");
	return 0;
}
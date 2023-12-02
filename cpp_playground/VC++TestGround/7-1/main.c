#include <stdio.h>

void sort_and_print_array(int *arr)
{
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2 - i; j++)
		{
			if (arr[j + 1] < arr[j])
			{
				int buffer = arr[j + 1];
				arr[j + 1] = arr[j];
				arr[j] = buffer;
			}
		}
	}

	for (int i = 0; i < 3; i++)
		printf("%d\t", arr[i]);
}

int main(void)
{
	int arr[3];
	for (int i = 0; i < 3; i++)
		scanf("%d", arr + i);

	int* p = arr;
	sort_and_print_array(p);
}
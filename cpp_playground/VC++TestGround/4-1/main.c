#include <stdio.h>

int main(void)
{
	int input[10];
	for (int i = 0; i < 10; i++)
		scanf("%d", input + i);

	// select sort
	int i, j, k, max, maxIndex;
	int count = 0;
	for (i = 0; i < 10; i++)
	{
		max = 0;
		maxIndex = 0;
		for (j = i; j < 10; j++)
		{
			if (input[j] > max)
			{
				max = input[j];
				maxIndex = j;
			}
		}

		// move
		for (k = maxIndex; k > count; k--)
			input[k] = input[k - 1];

		input[count++] = max;
	}

	// print
	for (int i = 0; i < 10; i++)
		printf("%d ", input[i]);

	return 0;
}

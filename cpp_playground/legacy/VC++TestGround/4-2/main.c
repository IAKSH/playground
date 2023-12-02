#include <stdio.h>

int main(void)
{
	int matrix[4][4];
	for (int i = 0; i < 4; i++)
	{
		printf("%d: ", i + 1);
		for (int j = 0; j < 4; j++)
			scanf("%d", &matrix[i][j]);
		puts("");
	}

	int row, line;
	int max = 0;
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if (matrix[i][j] > max)
			{
				max = matrix[i][j];
				row = j;
				line = i;
			}
		}
	}

	printf("max: %d\nrow: %d\nline: %d\n", max, row + 1, line + 1);
}

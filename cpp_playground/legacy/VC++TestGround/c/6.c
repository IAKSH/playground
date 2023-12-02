/*
#include <stdio.h>

int getNum(int m, int n)
{
	if (n == 1 || n == m || m == 1) return 1;
	return getNum(m - 1, n) + getNum(m - 1, n - 1);
}

int main()
{
	int arr[4][4];
	for (int i = 0; i < 4; i++)
		for(int j = 0;j < 4;j++)
			arr[i][j] = 0;

	for (int i = 1; i <= 4; i++)
	{
		for (int j = 1; j <= i; j++)
			arr[i - 1][j - 1] = getNum(i, j);
	}

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4 - i; j++)
			printf(" ");

		for (int j = 0; j < 4; j++)
		{
			if (arr[i][j] != 0)
				printf("%d ", arr[i][j]);
			else
			{
				puts("");
				break;
			}
		}
	}

	return 0;
}
*/
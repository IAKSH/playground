/*
#include <stdio.h>
#include <stdbool.h>

bool isPerfect(int n)
{
	int sum = 0;
	for (int i = 1; i < n; i++)
	{
		if (n % i == 0)
			sum += i;
	}

	return sum == n;
}

void printFactor(int n)
{
	for (int i = 1; i < n; i++)
	{
		if (n % i == 0)
			printf("%d有因子：%d\n", n, i);
	}
}

int main(void)
{
	for (int i = 1; i < 1000; i++)
	{
		if (isPerfect(i))
		{
			printf("找到完数：%d\n", i);
			printFactor(i);
		}
	}

	return 0;
}
*/
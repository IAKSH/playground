#include <stdio.h>

int main()
{
	int prime(int n);
	int n;
	scanf("%d", &n);
	if (prime(n))
		puts("������");
	else
		puts("��������");
	return 0;
}

int prime(int n)
{
	for (int i = 2; i < n; i++)
	{
		if ((n % i) == 0) 
			return 0;
	}
	return 1;
}
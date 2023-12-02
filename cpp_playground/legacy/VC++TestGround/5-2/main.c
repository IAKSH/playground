#include <stdio.h>

int gcd(int m, int n)
{
	if (m % n == 0) return n;
	return gcd(n, m % n);
}

int lcm(int m, int n)
{
	return m * n / gcd(m, n);
}

int main(void)
{
	int m, n;
	scanf("%d%d", &m, &n);
	printf("gcd=%d\nlcm=%d\n", gcd(m, n), lcm(m, n));
}
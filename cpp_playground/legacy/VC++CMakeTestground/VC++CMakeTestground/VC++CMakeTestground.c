#include <stdio.h>
#include <math.h>

void foo1()
{
	int x, y;
	printf("x = ");
	scanf("%d", &x);

	if (x < 1)
		y = x;
	else if (x >= 1 && x < 10)
		y = 2 * x - 1;
	else
		y = 3 * x - 11;

	printf("y = %d\n", y);
}

void foo2_if()
{
	int score;
	char mark;
	printf("score = ");
	scanf("%d", &score);
	if (score >= 90)
		mark = 'A';
	else if (score >= 80)
		mark = 'B';
	else if (score >= 70)
		mark = 'C';
	else if (score >= 60)
		mark = 'D';
	else
		mark = 'E';

	printf("level = %c\n", mark);
}

void foo2_switch()
{
	int score;
	char mark;
	printf("score = ");
	scanf("%d", &score);
	switch (score / 10)
	{
	case 9:
		mark = 'A';
		break;
	case 8:
		mark = 'B';
		break;
	case 7:
		mark = 'C';
		break;
	case 6:
		mark = 'D';
		break;
	default:
		mark = 'E';
	}

	printf("level = %c\n", mark);
}

void foo3()
{
	int n;
	scanf("%d", &n);

	int temp = n;
	int count = 0;
	while (temp > 0)
	{
		temp /= 10;
		count++;
	}
	printf("位数: %d\n", count);

	puts("正序");
	for (int i = 0; i < count; i++)
		printf("%d\n", (n / (int)(pow(10, i)) % 10));

	puts("倒序");
	for (int i = 0; i < count; i++)
		printf("%d\n", (n / (int)(pow(10, count - i - 1)) % 10));
}

int main()
{
	foo1();
	foo2_if();
	foo2_switch();
	foo3();
	return 0;
}

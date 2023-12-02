#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

// ц╟ещ
void sort(char* str,int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = i; j < n; j++)
		{
			if (str[i] < str[j])
			{
				char buffer = str[i];
				str[i] = str[j];
				str[j] = buffer;
			}
		}
	}

	str[n] = '\0';
	puts(str);
}

void sort1(char* str,int n)
{
	char max = CHAR_MIN;
	for (int i = 0; i < n; i++)
		if (str[i] > max)
			max = str[i];

	int* buffer = (int*)calloc(sizeof(char), max + 1);
	if (!buffer) abort;
	for (int i = 0; i < max + 1; i++)
		buffer[i] = 0;

	for (int i = 0; i < n; i++)
		buffer[str[i]] += 1;

	int index = 9;
	for (int i = 0; i < max + 1; i++)
	{
		while (buffer[i]-- > 0)
			str[index--] = i;

		if (index < 0)
			break;
	}

	str[n] = '\0';
	puts(str);
}

int main()
{
	char str[11];
	for (int i = 0; i < 10; i++)
		scanf("%c", str + i);
	sort1(str,10);
	return 0;
}
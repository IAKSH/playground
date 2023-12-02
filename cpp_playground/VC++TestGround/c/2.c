/*
#include <stdio.h>

#define MAX_INPUT 128

int main()
{
	char input[MAX_INPUT];
	int word = 0;
	int space = 0;
	int num = 0;
	int other = 0;

	gets(input);

	for (int i = 0; i < MAX_INPUT; i++)
	{
		if (input[i] == '\0')
			break;
		else if (input[i] == ' ')
			++space;
		else if (input[i] >= 48 && input[i] <= 57)
			++num;
		else if ((input[i] >= 65 && input[i] <= 90) || (input[i] >= 97 && input[i] <= 122))
			++word;
		else
			++other;
	}

	printf("字母：%d\n空格：%d\n数字：%d\n其他：%d\n", word, space, num, other);
	return 0;
}
*/
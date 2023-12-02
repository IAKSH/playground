#include <stdio.h>

int main()
{
	char str[3][80];
	for (int i = 0; i < 3; i++)
		gets(str[i]);

	int capitalLetter = 0;
	int lowercaseLetter = 0;
	int space = 0;
	int other = 0;

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 80; j++)
		{
			if (str[i][j] == '\0') break;
			else if (str[i][j] == ' ') space++;
			else if (str[i][j] >= 'A' && str[i][j] <= 'Z') capitalLetter++;
			else if (str[i][j] >= 'a' && str[i][j] <= 'z') lowercaseLetter++;
			else other++;
		}
	}

	printf("大写字母数：%d\n小写字母数：%d\n空格数：%d\n其他字符数：%d\n", capitalLetter, lowercaseLetter, space, other);
	return 0;
}

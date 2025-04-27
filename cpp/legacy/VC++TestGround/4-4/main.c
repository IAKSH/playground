#include <stdio.h>
#include <stdbool.h>

// won't check len
void strcat(char* dest, char* src)
{
	int destOriLen = 0;
	for (int i = 0; true; i++)
	{
		if (dest[i] == '\0')
		{
			destOriLen = i;
			break;
		}
	}

	for (int i = 0; true; i++)
	{
		if (src[i] == '\0')
			break;

		dest[destOriLen + i] = src[i];
	}
}

int main()
{
	char s0[32] = "c is ";
	char s1[16] = "strange";
	strcat(s0, s1);
	printf("%s\n", s0);
	return 0;
}

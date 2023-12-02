#include <stdio.h>
#include <limits.h>

float personal_average(float scores[5])
{
	float average = 0.0f;
	for (int i = 0; i < 5; i++)
		average += scores[i];

	return average / 5.0f;
}

float subject_average(float scores[10][5],unsigned int n)
{
	if (n >= 5)
		abort();

	float average = 0.0f;
	for (int i = 0; i < 10; i++)
		average += scores[i][n];

	return average / 10.0f;
}

float max_score(float* scores)
{
	float max = INT_MIN;
	for (int i = 0; i < 50; i++)
		if (scores[i] > max)
			max = scores[i];

	return max;
}

int main()
{
	float scores[10][5];
	for (int i = 0; i < 10; i++)
		for (int j = 0; j < 5; j++)
			//scores[i][j] = j + (j % 2);
			scanf("%f", &scores[i][j]);

	for (int i = 0; i < 10; i++)
		printf("第%d个人的平均分：%f\n", i, personal_average(&scores[i][0]));

	for (int i = 0; i < 5; i++)
		printf("第%d门课的平均分：%f\n", i, subject_average(scores,i));

	printf("最高分数：%f\n", max_score(scores));

	return 0;
}
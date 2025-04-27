#include <stdio.h>

#define STUDENTS_COUNT 6
#define SUBJECTS_COUNT 3

typedef struct
{
	int num;
	char name[16];
	float score[SUBJECTS_COUNT];
} Student;

static Student students[STUDENTS_COUNT];

void input(void)
{
	for (int i = 0; i < STUDENTS_COUNT; i++)
	{
		printf("输入学生%d的学号：", i);
		scanf("%d", &students[i].num);
		printf("输入学生%d的姓名：", i);
		scanf("%s", students[i].name);
		printf("输入学生%d的%d科分数：", i, SUBJECTS_COUNT);
		for (int j = 0; j < SUBJECTS_COUNT; j++)
			scanf("%f", students[i].score + j);
	}
}

void sort(void)
{
	for (int i = 0; i < STUDENTS_COUNT - 1; i++)
	{
		for (int j = 0; j < STUDENTS_COUNT - 1 - i; j++)
		{
			float sum_0 = 0.0f;
			float sum_1 = 0.0f;

			for (int k = 0; k < SUBJECTS_COUNT; k++)
				sum_0 += students[j].score[k];
			for (int k = 0; k < SUBJECTS_COUNT; k++)
				sum_1 += students[j + 1].score[k];

			if (sum_1 > sum_0)
			{
				Student buffer = students[j];
				students[j] = students[j+1];
				students[j + 1] = buffer;
			}
		}
	}

	for (int i = 0; i < STUDENTS_COUNT; i++)
	{
		printf("学生%d：\n学号：%d\n姓名：%s\n", i, students[i].num, students[i].name);
		for (int j = 0; j < SUBJECTS_COUNT; j++)
			printf("科目%d成绩：%f\n", j, students[i].score[j]);
		for (int j = 0; j < 16; j++)
			putchar('-');
		putchar('\n');
	}
}

void search(void)
{
	for (int i = 0; i < STUDENTS_COUNT; i++)
	{
		float sum = 0.0f;
		for (int j = 0; j < SUBJECTS_COUNT; j++)
			sum += students[i].score[j];

		if (sum > 270.0f)
		{
			printf("找到总成绩大于270分的学生：\n学号：%d\n姓名：%s\n", students[i].num, students[i].name);
			for (int j = 0; j < SUBJECTS_COUNT; j++)
				printf("科目%d成绩：%f\n", j, students[i].score[j]);
			for (int j = 0; j < 16; j++)
				putchar('-');
			putchar('\n');
		}
	}
}

int main(void)
{
	input();
	sort();
	search();
} 
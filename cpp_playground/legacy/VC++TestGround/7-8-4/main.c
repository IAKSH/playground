#include <stdio.h>
#include <stdlib.h>

typedef struct
{
    unsigned int number;
    char name[16];
    float scores[3];
}
Student;

static Student students[5];

void inputStu()
{
    for (int i = 0; i < 5; i++)
    {
        printf("student %d:\n", i);

        printf("number: ");
        scanf("%d", &students[i].number);

        printf("name: ");
        scanf("%s", students[i].name);


        for (int j = 0; j < 3; j++)
        {
            printf("score %d: ", j);
            scanf("%f", students[i].scores + j);
        }

        puts("");
    }
}

void search()
{
    int max_index = -1;
    float max_total_score = 0.0f;
    for (int i = 0; i < 5; i++)
    {
        float total_score = 0;
        for (int j = 0; j < 3; j++)
            total_score += students[i].scores[j];

        if (total_score > max_total_score)
        {
            max_total_score = total_score;
            max_index = i;
        }
    }

    Student* p = students + max_index;
    printf("学号：%d, 姓名：%s, 成绩1: %f, 成绩2: %f, 成绩3: %f, 总成绩: %f\n", p->number, p->name, p->scores[0], p->scores[1], p->scores[2], max_total_score);
}

int main(void)
{
    inputStu();
    search();
}
// https://www.lanqiao.cn/problems/107/learning/?page=1&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B&difficulty=10
// 抄的别人的题解
// 如果拿dp做好像会很难

#include <bits/stdc++.h>

using namespace std;

# define N 100010

//cnt[a]表示a积分的人数
int cnt[N];

int main() {
    int n, K, Max = 0;
    scanf("%d%d", &n,&K);
    for (int i = 0; i < n; ++i)
    {
        int a;
        scanf("%d", &a);
        cnt[a]++;
        Max = (Max > a) ? Max : a;
    }
    int math = 0;
    //k不等于0的情况
    for (int i = 0; i + K <= Max; ++i)
    {
        //i的积分和i+k的积分之差就等于k
        while (K && cnt[i] && cnt[i + K])
        {
            //有一组能匹配就++，直到其中一个积分的人数为0，然后进行下一组匹配
            math++;
            cnt[i]--;
            cnt[i + K]--;
        }
    }
    // k = 0的情况
    for (int i = 0; i <= Max; ++i)
    {
        while (!K && cnt[i] >= 2)
        {
            math += cnt[i] - 1;
            cnt[i] = 1;
        }
    }
    printf("%d", (n - math));

    return 0;
}
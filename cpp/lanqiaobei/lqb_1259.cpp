/**
 * https://www.lanqiao.cn/problems/1259/learning/?page=1&first_category_id=1&second_category_id=3&difficulty=30
 * 我觉得这道题有问题
 * 如果要尽量分的多的话，肯定是先从小的分起，这样就是i in range(0,7)，结果为35份
 * 但是题目答案是16，刚好就是i从7倒回0，也就是先从大的分起
*/

#include <bits/stdc++.h>

using namespace std;

#define ANSWER_16

int main() noexcept {
    int money = 1000000;
    int cnt = 0;

#ifdef ANSWER_16
    for(int i = 7;i >= 0;i--) {
#else
    bool finished = false;
    for(int i = 0;!finished;i++) {
#endif
        for(int j = 0;j < 5;j++) {
            int result = money - pow(7,i);
            if(result < 0) {
#ifndef ANSWER_16
                finished = true;
#endif
                break;
            }
            money = result;
            ++cnt;
        }
    }

    cout << cnt << '\n';
    return 0;
}
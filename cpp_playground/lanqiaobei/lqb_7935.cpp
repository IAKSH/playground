// https://www.lanqiao.cn/problems/7935/learning/?problem_list_id=25&page=1

#include <bits/stdc++.h>

using namespace std;

// 这道题和 big_num_stored_in_str.cpp 的情况不太一样
// 这里的是%1000，实际上就是取出2^2023的最后三位数

int main() noexcept {
    int n = 1;
    for(int i = 0;i < 2023;i++) {
        // 模拟了最后三维的增长情况，增长的位被忽略了
        n *= 2;
        n %= 1000;
    }
    cout << n << '\n';
    return 0;
}
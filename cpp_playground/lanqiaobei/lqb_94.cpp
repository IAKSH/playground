// https://www.lanqiao.cn/problems/94/learning/?page=1&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B
// N <= 100，好像可以暴力搜索
// nt蓝桥杯，我算是知道这题为什么是困难了
// 题干试图用卡片和票来迷惑人，卡片不是票，卡片上的标号表示这张卡能换多少票

#include <bits/stdc++.h>

using namespace std;

int n;
array<int,100> a;// real size = n
array<bool,100> flag;

bool check_empty() {
    for(int i = 0;i < n;i++) {
        if(flag[i])
            return false;
    }
    return true;
}

int main() {
    ios::sync_with_stdio(false);

    int i,j,cur,sum,max_sum = 0,max_val = 0;
    cin >> n;
    for(i = 0;i < n;i++) {
        cin >> a[i];
        max_val = max(max_val,a[i]);
    }

    //开始暴力搜索
    for(i = 0;i < n;i++) {
        sum = 0;
        cur = 0;
        fill(flag.begin(),flag.end(),true);
        for(j = 0;cur <= max_val && !check_empty();j++) {
            if(flag[(i + j) % n] && a[(i + j) % n] == ++cur) {
                flag[(i + j) % n] = false;
                sum += cur;
                cur = 0;
            }
        }
        max_sum = max(max_sum,sum);
    }

    cout << max_sum << '\n';
    return 0;
}
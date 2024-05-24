// https://www.lanqiao.cn/problems/691/learning/?page=2&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B&difficulty=10
// nmd, debug看见long long都爆了

#include <bits/stdc++.h>

using namespace std;

long long flip(long long x) {
    int res = 0;
    while(x > 0) {
        res = res * 10 + x % 10;
        x /= 10;
    }
    return res;
}

long long count(long long x) {
    long long res = 0;
    while(x > 0) {
        ++res;
        x /= 10;
    }
    return res;
}

bool check(long long x) {
    long long cnt = count(x);
    if(cnt == 1)
        return true;

    bool odd = !(cnt & 1LL);
    bool flag = false;
    stack<int> s;

    while(x > 0) {
        if(s.empty() || s.top() != x % 10) {
            if(!odd) {
                if(count(x) == cnt / 2 + 1) {
                    x /= 10;
                    continue;
                }
            }
            s.emplace(x % 10);
            x /= 10;
        }
        else {
            flag = true;
            s.pop();
            x /= 10;
        }
    }
    return s.empty();
}

int main() {
    ios::sync_with_stdio(false);
    int i,j;
    long long buf;
    for(i = 0;i < 200;i++) {
        buf = i;
        for(j = 0;!check(buf);j++) {
            // 一遍一遍调j上限，最后发现当j > 31时只有一个196的结果
            // 反正是填空题
            if(j > 31) {
                cout << i << '\n';
                //return 0;
            }
            buf = buf + flip(buf);
        }
    }
    return 0;
}
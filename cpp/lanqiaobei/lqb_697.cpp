// https://www.lanqiao.cn/problems/697/learning/?page=1&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B&difficulty=10&sort=pass_rate&asc=0

#include <bits/stdc++.h>

using namespace std;

bool check(int i,int buf) {
    unordered_map<char,int> m;
    while(i > 0) {
        ++m[i % 10];
        i /= 10;
    }
    while(buf > 0) {
        --m[buf % 10];
        buf /= 10;
    }
    for(const auto& p : m) {
        if(p.second != 0)
            return false;
    }
    return true;
}

int main() {
    array<int,5> a{2,3,4,5,6};
    int buf;
    bool flag;
    for(int i = 100000;i < 1000000;i++) {
        flag = true;
        for(const auto& j : a) {
            buf = i * j;
            if(buf < 100000 || buf >= 1000000 || !check(i,buf)) {
                flag = false;
                break;
            }
        }
        if(flag) {
            cout << i << '\n';
            return 0;
        }
    }
    return 1;
}
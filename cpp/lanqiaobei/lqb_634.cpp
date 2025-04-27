// https://www.lanqiao.cn/problems/634/learning/?page=1&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B&sort=pass_rate&asc=0
// 拿位权来做

#include <bits/stdc++.h>

using namespace std;

short trans(char c) {
    return static_cast<short>(c - 'A' + 10);
}

int trans(string s) {
    int len = s.size(),i = len,res = 0;
    while(i -- > 0) {
        res += trans(s[i]) * pow(36,len - i - 1);
    }
    return res;
}

int main() {
    cout << trans("MANY") << '\n';
    return 0;
}
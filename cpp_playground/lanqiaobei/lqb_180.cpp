/**
 * https://www.lanqiao.cn/problems/180/learning/?page=1&first_category_id=1&second_category_id=3&problem_id=180
 * 暴力，但是之前写的是三维vec，导致弄了很久也没弄出来
 * 题目上给了个地址映射公式，所以直接按着这个做就好了
*/

#include <bits/stdc++.h>

using namespace std;

static int a,b,c,m;

int getIndex(int i,int j,int k) noexcept {
    return ((i - 1) * b + (j - 1)) * c + (k - 1) + 1;
}

int main() noexcept {
    cin >> a >> b >> c >> m;
    vector<int> v(a * b * c);
    for(int i = 1;i <= a;i++) {
        for(int j = 1;j <= b;j++) {
            for(int k = 1;k <= c;k++) {
                cin >> v[getIndex(i,j,k)];
            }
        }
    }

    int lat,rat,lbt,rbt,lct,rct,ht;
    for(int dm = 0;dm < m;dm++) {
        cin >> lat >> rat >> lbt >> rbt >> lct >> rct >> ht;
        for(int i = lat;i <= rat && i <= a && i > 0;i++) {
            for(int j = lbt;j <= rbt && j <= b && j > 0;j++) {
                for(int k = lct;k <= rct && k <= c && k > 0;k++) {
                    if((v[getIndex(i,j,k)] -= ht) < 0) {
                        cout << dm + 1 << '\n';
                        return 0;
                    }
                }
            }
        }
    }
    cout << "-1\n";
    return 0;
}
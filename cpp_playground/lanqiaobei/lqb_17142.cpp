// https://www.lanqiao.cn/problems/17142/learning/?page=1&first_category_id=1&tags=%E5%9B%BD%E8%B5%9B,%E5%89%8D%E7%BC%80%E5%92%8C&sort=pass_rate&asc=0
// 还算是很简单的找规律，就是有个坑

#include <bits/stdc++.h>

using namespace std;

int main() {
    int n = 20230610;
    int res = 1;
    int layer = 1;
    int i;
    for(i = 2;res < n;i++) {
        layer += i;
        if(n - res >= layer)
            res += layer;
        else
            break;
    }
    // 因为break的时候i是在尝试搭建但是珠子不够的层，所以得-1来退回到最后一个能搭起来的层
    cout << i - 1 << '\n';
    return 0;
}
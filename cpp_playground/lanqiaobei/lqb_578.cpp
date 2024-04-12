/**
 * https://www.lanqiao.cn/problems/578/learning/?page=2&first_category_id=1&second_category_id=3&sort=pass_rate&asc=0
 * n个节点的无自环/重边的无向图有C(n,2)条边
 * 实际上就是一个排列组合问题，有公式 C(n,m) = (n!)/(m! * (n - m)!)
 * C(2020,2) = (2020!) / (2! * 2018!)
 * 2020!就是long long也撑不住，所以考虑从数学上化简
 * 将2020!和2018!展开，发现可以约分
 * 最后得到C(2022,2) = (2020 * 2019) / 2
*/
#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    cout << 2020 * 2019 / 2 << '\n';
    return 0;
}
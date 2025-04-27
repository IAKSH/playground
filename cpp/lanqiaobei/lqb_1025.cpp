// https://www.lanqiao.cn/problems/1025/learning/?page=1&first_category_id=1&tags=%E5%9B%BD%E8%B5%9B,%E8%B4%AA%E5%BF%83&sort=pass_rate&asc=0
// 比较中规中矩的贪心
// 就是会爆int，而且需要s+a+e排序，第一次弄成了s+a，没考虑e
// 100%

#include <bits/stdc++.h>

using namespace std;

struct N {
    long long s,a,e;
};

int main() {
    ios::sync_with_stdio(false);

    int n;cin >> n;
    vector<N> v(n);
    for(int i = 0;i < n;i++) {
        cin >> v[i].s >> v[i].a >> v[i].e;
    }

    // 先让快的来
    // 按s+a+e升序，然后逐个处理
    long long cnt = 0,cur = 0;
    sort(v.begin(),v.end(),[](const N& n1,const N& n2){return (n1.s + n1.a + n1.e) < (n2.s + n2.a + n2.e);});
    for(const auto& item : v) {
        cur += item.s + item.a;
        cnt += cur;
        cur += item.e;
    }

    cout << cnt << '\n';
    return 0;
}
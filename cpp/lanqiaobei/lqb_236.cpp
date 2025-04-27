// https://www.lanqiao.cn/problems/236/learning/?page=1&first_category_id=1&tags=%E5%9B%BD%E8%B5%9B,%E8%B4%AA%E5%BF%83&sort=pass_rate&asc=0
// 并查集？好像又不是
// 抄了别人的题解

/*
// [0]是整个树的根，不存储数据
using Arr = array<pair<int,N>,1001>;

int find(Arr& arr,int k) {
    if(arr[k].first == k)
        return k;
    return arr[k].first = find(arr,arr[k].first);
}

void merge(Arr& arr,int x,int y) {
    arr[find(arr,x)].first = find(arr,y);
}

bool check(Arr& arr,int x,int y) {
    return find(arr,x) == find(arr,y);
}
*/

#include <bits/stdc++.h>

using namespace std;

struct N {
    int x,y,v;
    short d;
};

const array<int,4> dx{0,0,-1,1};//up,down,left,right
const array<int,4> dy{1,-1,0,0};

int main() {
    ios::sync_with_stdio(false);

    int n,i,t,max_cnt = 0;cin >> n;
    array<N,1000> a;// real size = n

    char c;
    for(i = 1;i <= n;i++) {
        cin >> a[i].x >> a[i].y >> a[i].v >> c;
        switch(c) {
        case 'U':
            a[i].d = 0;
            break;
        case 'D':
            a[i].d = 1;
            break;
        case 'L':
            a[i].d = 2;
            break;
        case 'R':
            a[i].d = 3;
            break;
        }
    }

    unordered_map<int,int> mx;// mx[i] => x = i线上的点数
    unordered_map<int,int> my;// my[i] => y = i线上的点数

    for(t = 0; t < 1000; ++t) {
        mx.clear();
        my.clear();
        for(i = 1; i <= n; ++i) {
            // 还有这种操作？
            // 好吧，这种事情确实适合map来做，需要不定上限的下标
            // 就算不用map，也得自己实现一个类似的东西
            // 或者可以自己包装一个pair之类的，first存索引，second存数据。（等等，这不就是map吗？！）
            // 虽然这样的话可以用线性表来存，在一定数据量下也许会更快？
            // 但是语法上还是不如直接用map
            ++mx[a[i].x + a[i].v * t * dx[a[i].d]];
            ++my[a[i].y + a[i].v * t * dy[a[i].d]];
        }

        for(const auto& p : mx)
            max_cnt = max(max_cnt, p.second);
        for(const auto& p : my)
            max_cnt = max(max_cnt, p.second);
    }

    cout << max_cnt << '\n';
    return 0;
}
// https://www.luogu.com.cn/problem/P3367
// 递归查并集 + 路径压缩
// AC

// 一个还不错的题解
// https://www.luogu.com.cn/article/y83ljhmv

#include <bits/stdc++.h>

using namespace std;

// 按照树划分集合，然后递归查找根节点，并且顺便进行路径压缩
int find(int f[],int k) {
    // 根节点没有父节点，但是我们定义对于根节点有f[k] = k
    if(f[k] == k)
        return k;
    // 将当前节点的父节点改为直接指向所在树的根节点，所谓路径压缩。
    return f[k] = find(f,f[k]);
    // 不带路径压缩，会造成3个TLE
    //return find(f,f[k]);
}

int main() {
    int n,m,i,z,x,y;
    cin >> n >> m;

    // f[x]是x所在的树的父节点，通过递归能递归到所在树的根节点。
    int f[n];

    for(i = 1;i <= n;i++)
        // 初始化根节点的父指向自己
        f[i] = i;
    
    for(i = 1;i <= m;i++) {
        cin >> z >> x >> y;
        if(z == 1) {
            // 将x所在树的根节点直接改为y所在树的根节点，即完成了将x所在集合与y所在集合合并
            f[find(f,x)] = find(f,y);
        }
        else {
            // 直接通过比较x，y所在根节点是否相同来判断是否属于同一集合
            cout << (find(f,x) == find(f,y) ? 'Y' : 'N') << '\n';
        }
    }

    return 0;
}
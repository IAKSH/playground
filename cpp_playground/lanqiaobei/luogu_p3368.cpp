// https://www.luogu.com.cn/problem/P3368
// 由树状数组维护的差分数组 
// AC

#include <bits/stdc++.h>

using namespace std;

// 由树状数组维护的差分数组 
// 适合处理点查询和区间修改

class BITreeDiffArray {
public:
    BITreeDiffArray(vector<int>& v) {
        n = v.size();
        bitree.resize(n + 1, 0);
        for(int i = 0; i < n; i++)
            range_add(i + 1, i + 2, v[i]);
    }

    void range_add(int l, int r, int val) {
        add(l, val);
        add(r, -val);
    }

    int get_at(int i) {
        return prefix_sum(i);
    }

private:
    void add(int i, int val) {
        for(; i <= n; i += i & -i)
            bitree[i] += val;
    }

    int prefix_sum(int i) {
        int res = 0;
        for(; i > 0; i -= i & -i)
            res += bitree[i];
        return res;
    }

    vector<int> bitree;
    int n;
};

int main() {
    ios::sync_with_stdio(false);
    
    int n, m, a, b, c, d, i, j;
    cin >> n >> m;

    vector<int> v(n);
    for(auto& val : v)
        cin >> val;

    BITreeDiffArray bitree(v);
    
    for(i = 0; i < m; i++) {
        cin >> a;
        if(a == 1) {
            // 将区间 [b,c]内每个数加上d
            cin >> b >> c >> d;
            bitree.range_add(b, c + 1, d);
        }
        else {
            // 输出b处的值
            cin >> b;
            cout << bitree.get_at(b) << '\n';
        }
    }

    return 0;
}

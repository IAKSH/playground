// https://www.lanqiao.cn/problems/110/learning/?page=4&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B&sort=pass_rate&asc=0
// 并查集的简单应用
// 100%

#include <bits/stdc++.h>

using namespace std;

// real size = m * n + 1
// arr[i] => root of i
// arr[0] is root
array<int,1000001> arr;

int find_root(int i) {
    if(arr[i] == i)
        return i;
    // 路径压缩
    return arr[i] = find_root(arr[i]);
}

int main() {
    ios::sync_with_stdio(false);

    int m,n,k,a,b,i,root;
    cin >> m >> n >> k;
    
    for(int i = 0;i < m * n + 1;i++)
        arr[i] = i;

    while(k-- > 0) {
        cin >> a >> b;
        // merge
        arr[find_root(a)] = find_root(b);
    }

    // count
    unordered_set<int> set;
    for(i = 1;i < m * n + 1;i++) {
        root = find_root(arr[i]);
        if(set.find(root) == set.end())
            set.emplace(root);
    }
    cout << set.size() << '\n';
    return 0;
}
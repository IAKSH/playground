// https://www.luogu.com.cn/problem/P1177
// AC
// 非常的无脑，就算是纯c也能用qsort秒杀吧

#include <bits/stdc++.h>

using namespace std;

//#define USE_STDLIB_QSORT

#ifdef USE_STDLIB_QSORT
int comp(const void* n,const void* m) {
    return *(int*)n > *(int*)m;
}
#endif

int main() {
    ios::sync_with_stdio(false);
    int n;cin >> n;
    int arr[n];
    for(int i = 0;i < n;i++)
        cin >> arr[i];

#ifndef USE_STDLIB_QSORT
    sort(arr,arr + n);
#else
    qsort(arr,n,sizeof(int),comp);
#endif

    for(const auto& i : arr)
        cout << i << ' ';
    cout << '\n';
    return 0;
}
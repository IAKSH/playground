// https://www.lanqiao.cn/problems/2193/learning/?page=8&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B&sort=pass_rate&asc=0
// 实际上不用字符串（用字符串会非常低效）
// 解决方案是在dfs之前给n扩一位来进行保护，只要记录之前的位数以及计算max的时候减回来就行了
// 用了long long还是有几个爆了，而且快速幂都搬上来了还是有超时
// 看来得剪枝
// 50%

#include <bits/stdc++.h>

using namespace std;

int a,b,m;
long long n,offset,max_res = 0;

long long quick_power(long long a, long long b,long long c = LONG_LONG_MAX) {
	long long m = a,n = b,res = 1;
    while(n > 0) {
        if(n & 1)// 更快的n % 2 == 1
            res = res * m % c;
        m = m * m % c;
        n >>= 1;// 也许更快的n /= 2;
    }
    return res % c;
}

long long opA(long long x,int ind) {
    int val = x;
    for(int i = 0;i < ind;i++)
        val /= 10;
    val %= 10;
    if(val == 9)
        return x - 9 * quick_power(10,ind);
    else
        return x + quick_power(10,ind);
}

long long opB(long long x,int ind) {
    int val = x;
    for(int i = 0;i < ind;i++)
        val /= 10;
    val %= 10;
    if(val == 0)
        return x + 9 * quick_power(10,ind);
    else
        return x - quick_power(10,ind);
}

void dfs(long long x,int ind,bool op,int cur_a,int cur_b) {
    if(op)
        max_res = max(max_res,(x = opA(x,ind)) - offset);
    else
        max_res = max(max_res,(x = opB(x,ind)) - offset);
    for(int i = 0;i < m;i++) {
        if(cur_a)
            dfs(x,i,true,cur_a - 1,cur_b);
        if(cur_b)
            dfs(x,i,false,cur_a,cur_b - 1);
    }
}

int countDigits(int x) {
    int res = 0;
    while(x > 0) {
        x /= 10;
        ++res;
    }
    return res;
} 

int main() {
    ios::sync_with_stdio(false);
    cin >> n >> a >> b;
    n += (offset = quick_power(10,m = countDigits(n)));

    for(int i = 0;i < m;i++) {
        if(a)
            dfs(n,i,true,a - 1,b);
        if(b)
            dfs(n,i,false,a,b - 1);
    }

    cout << max_res << '\n';
    return 0;
}
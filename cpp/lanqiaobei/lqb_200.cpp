// https://www.lanqiao.cn/problems/200/learning/?page=5&first_category_id=1&second_category_id=3

#include <bits/stdc++.h>

using namespace std;

long long fib(int n) noexcept {
	long long fib1 = 1;
	long long fib2 = 0;
	if(n == 0)
		return fib2;
	if(n == 1)
		return fib1;
	int fib;
	for(int i = 2;i <= n;i++) {
		fib = fib1 + fib2;
		fib2 = fib1;
		fib1 = fib;
	}
	return fib;
}

int main() noexcept {
	int n;cin >> n;
    // 题目给的n范围巨大无比 (2x10^9)，十分容易溢出
    // 考虑寻找规律
    // 经过多次测试发现n >= 20以后所求的前八位小数不再变化
    // 题目上其实也暗示了：“F[N]/F[N + 1]，会趋近于黄金分割”
	if(n > 20)
		n = 20;
	printf("%.8f\n",static_cast<double>(fib(n)) / fib(n + 1)); 
	return 0;
} 
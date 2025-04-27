// https://leetcode.cn/problems/fibonacci-number/description/

#include <bits/stdc++.h>

using namespace std;

int fib(int n) noexcept {
	int fib1 = 1;
	int fib2 = 0;
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
	cout << fib(n) << '\n';
	return 0;
} 
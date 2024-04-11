// https://www.lanqiao.cn/problems/2366/learning/?page=6&first_category_id=1&second_category_id=3&sort=pass_rate&asc=0

#include <bits/stdc++.h>

using namespace std;

int factfact_last5(int n) noexcept {
	int res = 1;
	bool odd = n % 2;
	for(int i = 2;i <= n;i++) {
		if(i % 2 == odd) {
			res *= i;
			res %= 100000; 
		}
	}
	return res;
}

int main() noexcept {
	cout << factfact_last5(2021) << '\n';
	return 0;
}
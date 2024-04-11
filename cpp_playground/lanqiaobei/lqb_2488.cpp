// https://www.lanqiao.cn/problems/2488/learning/?page=7&first_category_id=1&second_category_id=3&sort=pass_rate&asc=0

#include <bits/stdc++.h>

using namespace std;

int lcm(int n) noexcept {
	return n * 2021 / __gcd(n,2021);
}

int main() noexcept {
	int cnt = 0;
	for(int i = 1;i <= 2021;i++) {
		cnt += (lcm(i) == 4042);
	}
	cout << cnt << '\n';
	return 0;
}
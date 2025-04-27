// https://www.lanqiao.cn/problems/593/learning/?page=9&first_category_id=1&second_category_id=3&sort=pass_rate&asc=0

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
	int cnt = 0;
	for(int i = 1;i <= 2020;i++) {
		for(int j = 1;j <= 2020;j++) {
			cnt += (__gcd(i,j) == 1);
		}
	}
	cout << cnt << '\n';
	return 0;
}
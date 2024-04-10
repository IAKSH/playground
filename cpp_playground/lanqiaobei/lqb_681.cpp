// https://www.lanqiao.cn/problems/681/learning/?page=2&first_category_id=1&second_category_id=3&sort=pass_rate&asc=0

// 最简单的第不知道多少集

#include <bits/stdc++.h>

using namespace std;

int acc(int n) noexcept {
	int res = 0;
	while(n > 0) {
		res += n % 10;
		n /= 10;
	}
	return res;
}

int main() noexcept {
	int cnt = 0;
    // 虽然一眼看不出右边界，但经过多次填入n个9的测试，发现自99以后直到99999 cnt都不再增长
	for(int i = 1;i < 99;i++) {
		cnt += (i == acc(i * i * i));
	}
	cout << cnt << '\n';
	return 0;
} 
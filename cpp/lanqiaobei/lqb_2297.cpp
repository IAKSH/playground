// https://www.lanqiao.cn/problems/2297/learning/?page=1&first_category_id=1&second_category_id=3&sort=pass_rate&asc=0

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
	int res = 92;
	for(int i = 1950;i < 2022;i++) {
		res += (((i % 4 == 0 && i % 100 != 0) || i% 400 == 0) ? 366 : 365);
	}
	cout << res << '\n';
	return 0;
}
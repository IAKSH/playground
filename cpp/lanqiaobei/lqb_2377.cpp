// https://www.lanqiao.cn/problems/2377/learning/?page=2&first_category_id=1&second_category_id=3&sort=pass_rate&asc=0

// 最简单的第?集

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
	int cnt = 0;
	string y;
	for(int i = 0;i < 5;i++) {
		cin >> y;
		cnt += (y[0] == y[2] && y[1] == y[3] - 1);
	}
	cout << cnt << '\n';
	return 0;
}
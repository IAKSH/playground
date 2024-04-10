// https://www.lanqiao.cn/problems/542/learning/?page=2&first_category_id=1&second_category_id=3&sort=pass_rate&asc=0

// 最无脑的第三集

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
	array<int,12> days{31,28,31,30,31,30,31,31,30,31,30,31};
	int n;cin >> n;
	cout << days[n - 1] << '\n';
	return 0;
}
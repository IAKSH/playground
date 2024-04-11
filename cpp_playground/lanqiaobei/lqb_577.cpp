// https://www.lanqiao.cn/problems/577/learning/?page=1&first_category_id=1&second_category_id=3&sort=pass_rate&asc=0

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
	int cnt = 0;
	int n;
	for(int i = 1;i <= 2020;i++) {
		n = i;
		while(n > 0) {
			++cnt;
			n /= 10;	
		}
	}
	cout << cnt << '\n';
	return 0;
}
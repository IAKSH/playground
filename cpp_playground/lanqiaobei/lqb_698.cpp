// https://www.lanqiao.cn/problems/698/learning/?page=2&first_category_id=1&second_category_id=3&sort=pass_rate&asc=0

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
	for(int i = 1;true;i++) {
		for(int j = 0;j < 8;j++) {
			if(i * (i + j) == 6 * (2 * i + j)) {
				cout << i << '\n';
				return 0;
			}
		}
	}
	return 1;
}
// https://www.lanqiao.cn/problems/2140/learning/?page=10&first_category_id=1&second_category_id=3&sort=pass_rate&asc=0

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
	int day = 6;
	int offset = 20 % 8;
    // 虽然我并没有理解为什么是<=22
	for(int i = 0;i <= 22;i++) {
		day += offset;
		if(day > 7)
			day -= 7;
		//day %= 8;
	}
	cout << day << '\n';
	return 0;
}
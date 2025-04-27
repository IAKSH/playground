// https://www.lanqiao.cn/problems/2296/learning/?page=2&first_category_id=1&second_category_id=3&sort=pass_rate&asc=0

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    // 由于只涉及了低六位，不需要使用unsigned
	for(int i = 2022;true;i++) {
		if((i & 0x0000003f) == 0) {
			cout << i << '\n';
			break;
		}
	}
	return 0;
}
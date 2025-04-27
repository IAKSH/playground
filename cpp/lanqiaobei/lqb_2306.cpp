// https://www.lanqiao.cn/problems/2306/learning/?page=2&first_category_id=1&second_category_id=3&tags=2022

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
	int x = 0;
	int y = 0;
	string s;cin >> s;
	for(const auto& c : s) {
		switch(c) {
		case 'U': --x;break;
		case 'D': ++x;break;
		case 'L': --y;break;
		case 'R': ++y;break;
		}
	}
	cout << x << ' ' << y << '\n';
	return 0;
}
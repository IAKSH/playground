// https://www.lanqiao.cn/problems/2293/learning/?page=5&first_category_id=1&second_category_id=3&sort=pass_rate&asc=0

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
	string s;cin >> s;
	int cnt = 0;
	for(int i = s.size() - 1;i >= 0;i--) {
		if(s[i] == '0')
			++cnt;
		else
			break;
	}
	cout << cnt << '\n';
	return 0;
}
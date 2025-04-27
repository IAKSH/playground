// https://www.lanqiao.cn/problems/1452/learning/?page=10&first_category_id=1&second_category_id=3&sort=pass_rate&asc=0
// 其实也许可以直接拿chrono做

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
	long long n;cin >> n;
	n /= 1000;
	int h = n / 3600 % 24;
	int m = n % 3600 / 60;
	int s = n % 60;
	if(h < 10) cout << '0' << h << ':';
	else cout << h << ':';
	if(m < 10) cout << '0' << m << ':';
	else cout << m << ':';
	if(s < 10) cout << '0' << s << '\n';
	else cout << s << '\n';
	return 0;
}
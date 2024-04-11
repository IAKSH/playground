// https://www.lanqiao.cn/problems/2353/learning/?page=10&first_category_id=1&second_category_id=3&sort=pass_rate&asc=0

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
	int n,m;cin >> n >> m;
	array<array<char,4>,12> arr{
	"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"
	};
	cout << arr[n - 1].data();
	if(m < 10) cout << '0';
	cout << m << '\n';
	return 0;
}
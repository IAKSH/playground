// https://www.lanqiao.cn/problems/543/learning/?page=2&first_category_id=1&second_category_id=3&sort=pass_rate&asc=0

// 最无脑的第四集

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
	int n;cin >> n;
	vector<int> v(n);
	for(auto& i : v)
		cin >> i;
		
	int maxn = INT_MIN;
	for(int i = 0;i < n - 1;i++) {
		maxn = max(maxn,v[i + 1] - v[i]);
	}
	cout << maxn << '\n';
	return 0;
}
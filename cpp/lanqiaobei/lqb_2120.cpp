// https://www.lanqiao.cn/problems/2120/learning/?page=8&first_category_id=1&second_category_id=3&sort=pass_rate&asc=0

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
	int w = 1189;
	int h = 841;
	array<pair<int,int>,10> arr;
	arr[0].first = 1189;
	arr[0].second = 841;
	for(int i = 1;i < 10;i++) {
		if(arr[i - 1].first > arr[i - 1].second) {
			int a = arr[i - 1].first / 2;
			int b = arr[i - 1].second;
			arr[i].first = max(a,b);
			arr[i].second = min(a,b);
		}
		else {
			int a = arr[i - 1].first;
			int b = arr[i - 1].second / 2;
			arr[i].first = max(a,b);
			arr[i].second = min(a,b);
		}
	}
	string s;cin >> s;
	auto p = arr[s[1] - '0'];
	cout << p.first << '\n' << p.second << '\n';
	return 0; 
}
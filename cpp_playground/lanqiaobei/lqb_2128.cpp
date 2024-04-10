// https://www.lanqiao.cn/problems/2128/learning/?page=2&first_category_id=1&second_category_id=3&tags=2022
// 60% 1错3超时

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
	int n;cin >> n;
	vector<int> v(n);
	for(auto& i : v)
		cin >> i;
	int m;cin >> m;
	vector<pair<int,int>> sel_v(m); 
	for(auto& p : sel_v)
		cin >> p.first >> p.second;
	
	long long ori_sum = 0;
	vector<int> cnt_v(100001,0);
	int cnt_v_min = INT_MAX;
	int cnt_v_max = INT_MIN;
	for(const auto& p : sel_v) {
		ori_sum += accumulate(v.begin() + p.first - 1,v.begin() + p.second,0);
		cnt_v_min = min(cnt_v_min,p.first);
		cnt_v_max = max(cnt_v_max,p.second);
		for(int i = p.first;i <= p.second;i++) {
			++cnt_v[i];
		}
	}
	
	sort(v.begin(),v.end(),greater<int>());
	long long sum = 0;
	for(const auto& i : v) {
		vector<int>::iterator it = max_element(cnt_v.begin(),cnt_v.end());
		while(*it != 0) {
			--(*it);
			sum += i;
		} 
	}
	cout << sum - ori_sum << '\n';
	return 0;
}
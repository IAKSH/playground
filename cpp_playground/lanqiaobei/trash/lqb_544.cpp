// https://www.lanqiao.cn/problems/544/learning/?page=1&first_category_id=1&second_category_id=3&tags=%E5%AD%97%E7%AC%A6%E4%B8%B2

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    string s;cin >> s;
    int t;cin >> t;
    
    for(int i = 0;i < t;i++) {
    	s.erase(max_element(s.begin(),s.end()) - s.begin(),1);
	}
	
	cout << s << '\n';
	return 0; 
}
// https://www.lanqiao.cn/problems/2119/learning/?page=1&first_category_id=1&second_category_id=3&tags=2022
// bad, answer should be 212
// abandoned

#include <bits/stdc++.h>

using namespace std;

bool check(array<int,12>::iterator it) noexcept {
	int a = 0;
	int b = 0;
	int a_cnt = 0;
	int b_cnt = 0;
	bool a_recorded = false;
	bool b_recorded = false;
	//for(;it != it + 4;it++) {
	for(int i = 0;i < 4;i++) {
		if(!a_recorded) {
			a = *(it + i);
			++a_cnt;
			a_recorded = true;
		}
		else if(!b_recorded && a != *(it + i)) {
			b = *(it + i);
			++b_cnt;
			b_recorded = true;
		}
		else if(a == *(it + i)) {
			++a_cnt;
			if(b > 1 && a > 1)
				return false;
		}
		else if(b == *(it + i)) {
			++b_cnt;
			if(b > 1 && a > 1)
				return false;
		}
		else
			return false;
	}
	return (a_recorded && b_recorded && max(a_cnt,b_cnt) == 3 && min(a_cnt,b_cnt) == 1);
}

int main() noexcept {
	array<int,12> days{31,28,31,30,31,30,31,31,30,31,30,31};
	array<int,12> date;
	int cnt = 0;
	for(int i = 1111;i <= 9999;i++) {
		date[0] = i / 1000;
		date[1] = i / 100 % 10;
		date[2] = i / 10 % 10;
		date[3] = i % 10;
		// check year
		if(!check(date.begin()))
			continue;
		for(int j = 1;j <= 12;j++) {
			date[4] = j / 10;
			date[5] = j % 10;
			for(int k = 1;k <= (j == 2 ? (((i % 4 == 0 && i % 100 != 0) || i % 400 == 0) ? 29 : 28) : days[j]);k++) {
				date[6] = k / 10;
				date[7] = k % 10;
				// check month & date
				if(!check(date.begin() + 4))
					continue;
				for(int l = 0;l < 24;l++) {
					date[8] = l / 10;
					date[9] = l % 10;
					for(int m = 0;m < 60;m++) {
						date[10] = m / 10;
						date[11] = m % 10;
						// check time
						if(check(date.begin() + 8)) {
                            cout << "\n---\n";
                            for(int n = 0;n < 12;n++) {
                                if(n != 0 && n % 4 == 0)
                                    cout << '\n';
                                cout << date[n];
                            }
                            ++cnt;	
                        }
					}
				} 
			}
		}
	}
	cout << cnt << '\n';
	return 0;
}
/**
 * https://www.lanqiao.cn/problems/3493/learning/?page=1&first_category_id=1&second_category_id=3&difficulty=20&tags=2023
 * accumulate 1..20230408
 * 实际上是一个等差数列
 * a_1 = 1
 * a_n = 20230408
 * S_n = a * a_1 + (n * (n - 1) * d) / 2
 *     = n * (a_1 + a_n) / 2
*/

#include <bits/stdc++.h>

using namespace std;

#ifdef USE_ARITHMETIC_SEQUENCE
int main() noexcept {
    long long sum = 0;
    sum = (1 + 20230408LL) * 20230408LL / 2;
    cout << sum;
    return 0;
}
#else
int main() noexcept {
	long long res = 0;
	for(int i = 1;i <= 20230408;i++) {
		res += i;
	}
	cout << res << '\n';
	return 0;
} 
#endif
/*
#define USING_STRING_ACCUMULATE

string strAccumulate(string a,string b) noexcept {
    reverse(a.begin(),a.end());
    reverse(b.begin(),b.end());
    string result_str;
    string& shorter = (a.size() < b.size() ? a : b);
    string& longger = (shorter == a ? b : a);

    bool carry = false;
    int shorter_len = shorter.size();
    int i = 0;
    for(;i < shorter_len;i++) {
        int res = shorter[i] - '0' + longger[i] - '0' + carry;
        if(res >= 10) {
            carry = true;
            res -= 10;
        }
        else
            carry = false;
        result_str.push_back(static_cast<char>(res + '0'));
    };
    int longger_len = longger.size();
    for(;i < longger_len;i++) {
        int tmp = longger[i] + carry;
        if(tmp > 10)
            tmp -= 10;
        else
            carry = false;
        result_str.push_back(static_cast<char>(tmp));
    }
    if(carry)
        result_str.push_back('1');
    reverse(result_str.begin(),result_str.end());
    return result_str;
}

int main() noexcept {
#ifndef USING_STRING_ACCUMULATE
    long long sum = 0;
    sum = (1 + 20230408LL) * 20230408LL / 2;
    cout << sum;
#else
    string res = "0";
    for(int i = 1;i <= 20230408;i++)
        res = strAccumulate(res,to_string(i));
    cout << res << '\n';
#endif
    return 0;
}
*/
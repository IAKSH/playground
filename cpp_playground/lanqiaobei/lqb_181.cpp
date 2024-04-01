/**
 * https://www.lanqiao.cn/problems/181/learning/?page=2&first_category_id=1&second_category_id=3&difficulty=20
 * 题目上有提到说输入的是字符串，原来是因为测试例里全是些能把int干穿的
 * 甚至long long也干穿了，只过了一个
*/

#ifdef USING_LONG_LONG
#include <bits/stdc++.h>

using namespace std;

int superSum(long long n) noexcept {
    if(n < 10) return n;
    int sum = 0;
    while(n > 0) {
        sum += n % 10;
        n /= 10;
    }
    return superSum(sum);
}

int main() noexcept {
    long long n;cin >> n;
    cout << superSum(n) << '\n';
    return 0;
}
#else
#include <bits/stdc++.h>

using namespace std;

int superSum(long long n) noexcept {
    if(n < 10) return n;
    int sum = 0;
    while(n > 0) {
        sum += n % 10;
        n /= 10;
    }
    return superSum(sum);
}

int main() noexcept {
    string s;cin >> s;
    
    int sum = 0;
    for(const auto& c : s)
        sum += c - '0';
    
    cout << superSum(sum) << '\n';
    return 0;
}
#endif
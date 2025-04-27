// https://www.lanqiao.cn/problems/160/learning/?page=1&first_category_id=1&second_category_id=3&difficulty=20

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    array<char,5> vowel{'a','e','i','o','u'};
    string s;cin >> s;
    int cnt_vowel = 0;
    int cnt_others = 0;
    for(const auto& c : s)
        ++(find(vowel.begin(),vowel.end(),c) != vowel.end() ? cnt_vowel : cnt_others);
    cout << cnt_vowel << '\n' << cnt_others << '\n';
    return 0;
}
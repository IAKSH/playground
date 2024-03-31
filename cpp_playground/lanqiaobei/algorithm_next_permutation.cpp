// https://www.lanqiao.cn/problems/269/learning/?page=1&first_category_id=1&second_category_id=3&tags=%E5%85%A8%E6%8E%92%E5%88%97
// next_permutation，（对于一个原本有序的range，是否存在）下一个（全）排列

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    string input;
    cin >> input;
    int len = input.size();
    array<char,10> chars{'a','b','c','d','e','f','g','h','i','j'};
    //sort(chars.begin(),chars.begin() + n);

    int cnt = 0;
    stringstream ss;
    do {
        ss.str("");
        ss.clear();
        for(int i = 0;i < len;i++) {
            ss << chars[i];
        }
        //cout << ss.str() << '\n';
        if(ss.str() == input)
            break;
        ++cnt;
    } while (next_permutation(chars.begin(),chars.begin() + len));// magic

    cout << cnt << '\n';
    return 0;
}
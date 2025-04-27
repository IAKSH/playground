// https://www.lanqiao.cn/problems/269/learning/?page=1&first_category_id=1&second_category_id=3&tags=%E5%85%A8%E6%8E%92%E5%88%97

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

/*
int main() noexcept {
    test();
    return 0;

    // 可以全排列为一个10层的树
    // 通过画图可以得到规律，然后用代码实现之
    array<char,10> chars{'a','b','c','d','e','f','g','h','i','j'};
    
    string input;
    cin >> input;
    int len = input.size();

    int sum = 0;
    int last_layer_val = 0;
    for(int i = 0;i < len;i++) {
        int this_layer_val = find(chars.begin(),chars.end(),input[i]) - chars.begin();
        sum += this_layer_val + (len - 1 - i) * last_layer_val;
        last_layer_val += this_layer_val;
        remove(chars.begin(),chars.end(),input[i]);
    }

    cout << sum << '\n';
    return 0;
}
*/
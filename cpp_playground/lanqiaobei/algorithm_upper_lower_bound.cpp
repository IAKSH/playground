#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

template <typename T>
void println_address(T& t) {
    for(auto it = begin(t);it != end(t);it++) {
        cout << &*it << ' ';
    }
    cout << '\n';
}

int main() {
    vector<int> v = {1, 2, 4, 4, 5, 6};
    println_address(v);

    // 查找第一个大于4的元素
    auto upper_it = std::upper_bound(v.begin(), v.end(), 4);
    // 查找第一个大于或等于4的元素
    auto lower_it = std::lower_bound(v.begin(), v.end(), 4);

    cout << "The upper bound of 4 in the vector is: " << *upper_it << " at " << &*upper_it << endl;
    cout << "The lower bound of 4 in the vector is: " << *lower_it << " at " << &*lower_it << endl;

    return 0;
}

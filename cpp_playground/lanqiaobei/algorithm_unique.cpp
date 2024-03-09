#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

/**
 * 用于移除范围 [first, last) 中连续重复的元素。
 * 它并不删除所有的重复元素，而是通过将这些元素替换为序列中下一个不重复的元素来移除重复性。
 * 所有被替换的元素都处于未指定的状态。
*/

int main () {
    vector<int> v = { 1, 1, 3, 3, 3, 10, 1, 3, 3, 7, 7, 8 };
    vector<int>::iterator ip;

    cout << "size before std::unique() = " << v.size() << '\n';
    ip = std::unique (v.begin (), v.begin () + 12);
    cout << "size after std::unique() = " << v.size() << '\n';
    v.resize (std::distance (v.begin (), ip));
    cout << "size after resize = " << v.size() << '\n';

    for (ip = v.begin (); ip != v.end (); ++ip) {
        cout << *ip << " ";
    }

    return 0;
}

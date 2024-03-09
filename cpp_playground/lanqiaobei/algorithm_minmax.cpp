#include<iostream>
#include<algorithm>

using namespace std;

/**
 * std::minmax (C++ 11)
 * 用于找出两个值中的最大值和最小值
*/

int main() {
    pair<int, int> mnmx;

    mnmx = minmax(53, 23);

    cout << "The minimum value obtained is : ";
    cout << mnmx.first;
    cout << "\nThe maximum value obtained is : ";
    cout << mnmx.second;

    mnmx = minmax({2, 5, 1, 6, 3});

    cout << "\n\nThe minimum value obtained is : ";
    cout << mnmx.first;
    cout << "\nThe maximum value obtained is : ";
    cout << mnmx.second;

    return 0;
}

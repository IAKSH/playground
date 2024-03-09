#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

/**
 * std::unique_copy
 * 详见algorithm_unique.cpp
*/

int main () {
    vector<int> v = { 10, 10, 30, 30, 30, 100, 10, 300, 300, 70, 70, 80 };
    vector<int> v1 (10);
    vector<int>::iterator ip;

    ip = std::unique_copy (v.begin (), v.begin () + 12, v1.begin ());
    v1.resize (std::distance (v1.begin (), ip));

    cout << "Before: ";
    for (ip = v.begin (); ip != v.end (); ++ip) {
        cout << *ip << " ";
    }
    cout << "\nAfter: ";
    for (ip = v1.begin (); ip != v1.end (); ++ip) {
        cout << *ip << " ";
    }

    return 0;
}

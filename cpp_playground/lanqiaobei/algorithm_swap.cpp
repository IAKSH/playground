#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

using namespace std;


int main() noexcept {
    int arr1[13] = {0,1,2,3,4,5,6,3,4,-1,-2,-3,-4};
    int arr2[13] = {0};

    for(const auto& i : arr1) {
        cout << i << ' ';
    }
    cout << '\n';
    for(const auto& i : arr2) {
        cout << i << ' ';
    }
    cout << '\n';

    swap(arr1,arr2);
    cout << "swap\n";

    for(const auto& i : arr1) {
        cout << i << ' ';
    }
    cout << '\n';
    for(const auto& i : arr2) {
        cout << i << ' ';
    }
    cout << '\n';

    return 0;
}
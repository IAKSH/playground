#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

int main() noexcept {
    int arr1[9];
    int arr2[3];
    vector<int> v2{1,2,3,4,5,6,7,8,9};
    copy(v2.begin(),v2.end(),begin(arr1));
    for(const auto& i : arr1) {
        cout << i << '\n';
    }

    cout << "arr2:\n";
    copy(begin(arr1),begin(arr1) + 3,begin(arr2));
    for(auto it = begin(arr2);it != end(arr2);it++) {
        cout << *it << '\n';
    }
    return 0;
}
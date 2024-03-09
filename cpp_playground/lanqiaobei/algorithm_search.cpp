#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

using namespace std;


int main() noexcept {
    int arr1[] = {0,1,2,3,4,5,6,3,4,-1,-2,-3,-4};
    int arr2[] = {3,4};

    for(auto it = search(begin(arr1),end(arr1),begin(arr2),end(arr2));it != end(arr1);it++) {
        cout << *it << '\n';
    }

    return 0;
}
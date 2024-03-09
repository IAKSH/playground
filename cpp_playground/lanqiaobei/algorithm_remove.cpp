#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

using namespace std;

int main() noexcept {
    int arr1[13] = {0,1,2,3,4,5,6,3,4,-1,-2,-3,-4};

    remove(begin(arr1),end(arr1),2);
    remove(begin(arr1),end(arr1),3);
    remove(begin(arr1),end(arr1),4);
    remove(begin(arr1),end(arr1),5);
    
    for(const auto& i : arr1) {
        cout << i << ' ';
    }
    cout << endl;

    return 0;
}
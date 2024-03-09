#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

using namespace std;

int main() noexcept {
    int arr1[] = {1,2,3,4,5,6};
    int arr2[] = {1,2,3,4,5,6};
    int arr3[] = {1,2,3,3,5,6};

    cout << equal(begin(arr1),end(arr1),begin(arr2)) << endl;
    cout << equal(begin(arr1),end(arr1),begin(arr3)) << endl;

    return 0;
}
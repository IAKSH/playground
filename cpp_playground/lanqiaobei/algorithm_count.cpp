#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

constexpr int ARR_LEN = 514;

int main() noexcept {
    int cnt = 0;
    int arr[ARR_LEN];
    for(auto it = begin(arr);it != end(arr);it++) {
        *it = cnt++;
    }

    cout << count(arr,arr + ARR_LEN,114) << '\n';
    return 0;
}
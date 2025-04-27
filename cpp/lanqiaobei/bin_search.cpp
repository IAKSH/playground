#include <iostream>

static int arr[]{-1,0,1,4,6,9,14,65,114,514,999};

bool check(int i) noexcept {
    return arr[i] < 11;
}

int main() noexcept {
    int l = 0;
    int r = sizeof(arr) / sizeof(int);
    int mid;
    while(l != r) {
        mid = (l + r) / 2;
        if(check(mid))
            l = mid + 1;
        else
            r = mid;
    }
    std::cout << arr[l - 1] << '\n';
    return 0;
}
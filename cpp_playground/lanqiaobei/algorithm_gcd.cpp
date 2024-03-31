#include <iostream>
#include <algorithm>

int main() noexcept {
    int m,n;
    std::cin >> m >> n;
    std::cout << std::__gcd(m,n) << '\n';
    return 0;
}
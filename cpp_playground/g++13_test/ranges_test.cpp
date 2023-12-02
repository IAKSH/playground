#include <iostream>
#include <format>
#include <ranges>

bool is_prime(int n) {
    for (int i = 1;i < n;i++) {
        if (n % i == 0)
            return false;
    }
    return true;
}

int main() noexcept {
    for (int prime : std::ranges::iota_view(1) | std::ranges::filter(is_prime) | std::ranges::take(50)) {
        std::cout << std::format("find prime {} !",prime);
    }
}
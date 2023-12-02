#include <iostream>
#include <format>
#include <ranges>

bool oushu(int x) noexcept {
	return x % 2 == 0;
}

int main() noexcept {
	for (int x : std::ranges::iota_view(1) | std::views::take(10) | std::views::filter(oushu)) {
		std::cout << std::format("oushu find: {}\n", x);
	}
}
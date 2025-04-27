#include <variant>
#include <iostream>

std::variant<float, bool> get_result(std::variant<int, char> arg) {
    return std::visit([](auto&& value) -> std::variant<float, bool> {
        using T = std::decay_t<decltype(value)>;

        if constexpr (std::is_same_v<T, int>) {
            return 3.14f;
        } else if constexpr (std::is_same_v<T, char>) {
            return true;
        }
    }, arg);
}

int main() {
    auto result1 = get_result(42);
    std::cout << "Result for int input: " << std::get<float>(result1) << '\n';

    auto result2 = get_result('A');
    std::cout << "Result for char input: " << std::get<bool>(result2) << '\n';

    return 0;
}
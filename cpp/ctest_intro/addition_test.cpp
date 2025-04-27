#include <iostream>

int add(int a, int b) {
    return a + b;
}

int main() {
    int result = add(2, 3);
    if (result == 5) {
        std::cout << "Test passed: 2 + 3 = " << result << '\n';
        return 0;
    } else {
        std::cout << "Test failed: 2 + 3 != " << result << '\n';
        return 1;
    }
}

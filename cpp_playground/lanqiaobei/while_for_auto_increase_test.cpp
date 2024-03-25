#include <iostream>

/**
 * for中的i++和++i行为一致，统一按照i++
 * while中的i++和++i符合常理
*/

void testWhile1() noexcept {
    int i = 4;
    while(i-- > 0) {
        std::cout << i << std::endl;
    }
    std::cout << "after while(i-- > 0): " << i << std::endl;
}

void testWhile2() noexcept {
    int i = 4;
    while(--i > 0) {
        std::cout << i << std::endl;
    }
    std::cout << "after while(--i > 0): " << i << std::endl;
}

void testFor1() noexcept {
    int i = 4;
    for(;i > 0;i--) {
        std::cout << i << std::endl;
    }
    std::cout << "after for(;i > 0;i--): " << i << std::endl;
}

void testFor2() noexcept {
    int i = 4;
    for(;i > 0;--i) {
        std::cout << i << std::endl;
    }
    std::cout << "after for(;i > 0;--i): " << i << std::endl;
}

int main() noexcept {
    testWhile1();
    testWhile2();
    testFor1();
    testFor2();
    return 0;
}
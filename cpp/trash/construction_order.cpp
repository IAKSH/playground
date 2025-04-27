#include <iostream>
#include <string_view>

class Base {
public:
    Base(std::string_view name) noexcept {
        std::cout << "Base name = " << name << std::endl;
    }

    ~Base() noexcept {
        std::cout << "~Base" << std::endl;
    }
};

class Son : public Base {
public:
    Son() noexcept 
        : Base("idk") {
        std::cout << "Son" << std::endl;
    }

    ~Son() noexcept {
        std::cout << "~Son" << std::endl;
    }
};

int main() noexcept {
    Son s;
    return 0;
}
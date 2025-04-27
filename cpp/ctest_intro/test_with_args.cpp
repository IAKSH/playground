#include <iostream>
#include <format>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << std::format("Usage: {} <argument>\n",argv[0]);
        return 1;
    }
    if (std::string(argv[1]) == "pass") {
        std::cout << "Test Passed\n";
        return 0;
    } else {
        std::cout << "Test failed!\n";
        return 1;
    }
}

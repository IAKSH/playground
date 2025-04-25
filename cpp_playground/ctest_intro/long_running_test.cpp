#include <iostream>
#include <thread>
#include <chrono>

int main() {
    std::cout << "Starting long-running test..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(10));
    std::cout << "Test completed!" << std::endl;
    return 0;
}

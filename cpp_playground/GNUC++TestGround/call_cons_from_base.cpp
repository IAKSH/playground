#include <iostream>

class Base
{
public:
    Base(float f) {std::cout << f;}
    ~Base() = default;
};

struct Diverse : public Base
{
    Diverse(float f) : Base(f) {}
};

int main()
{
    Diverse d(114514);
}
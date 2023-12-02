#include <iostream>

template <typename T>
struct Base
{
    Base(int i) {std::cout << "fuck you!\n" << "i = " << i << std::endl;};
};

struct Son : Base<Son>
{
    Son(int i) : Base(i)
    {};
};

int main()
{
    Son s(114);
    return 0;
}
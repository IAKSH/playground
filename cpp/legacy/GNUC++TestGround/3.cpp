#include <iostream>

int main()
{
    int b,a;
    b = (a = 2 + 3,a*4),a + 5;
    std::cout << a << ',' << b << std::endl;
}
#include <iostream>

class Container
{
private:
    inline static int val;
public:
    Container()
    {

    }
    ~Container() = default;
    int& getVal()
    {
        return val;
    }
};

int main() noexcept
{
    Container c1,c2;
    c1.getVal() = 114514;
    c2.getVal() = 1919810;
    std::cout << c1.getVal() << std::endl;
    return 0;
}
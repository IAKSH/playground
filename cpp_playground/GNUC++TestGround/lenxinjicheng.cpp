#include <iostream>

class BaseA
{
public:
    BaseA()
    {
        std::cout << "BaseA()\n";
    }
    ~BaseA()
    {
        std::cout << "~BaseA()\n";
    }
};

class SonA : virtual public BaseA
{
public:
    SonA()
    {
        std::cout << "SonA()\n";
    }
    ~SonA()
    {
        std::cout << "~SonA()\n";
    }
};

class SonB : virtual public BaseA
{
public:
    SonB()
    {
        std::cout << "SonB()\n";
    }
    ~SonB()
    {
        std::cout << "~SonB()\n";
    }
};

class Top : public SonA, public SonB
{
public:
    Top()
    {
        std::cout << "Top()\n";
    }
    ~Top()
    {
        std::cout << "~Top()\n";
    }
};

int main() noexcept
{
    Top t;
    return 0;
}
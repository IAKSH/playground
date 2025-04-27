#include <iostream>

class Base
{
protected:
    Base() = default;

public:
    virtual ~Base() = default;
};

class Div : public Base
{
public:
    Div() = default;
    virtual ~Div() override = default;
};

int main()
{
    Base b;
    Div d;
    return 0;
}
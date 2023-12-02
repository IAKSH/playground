#include <iostream>

template <typename T>
struct Base
{
    void doit() {static_cast<T*>(this)->imp_doit();}
};

class Diverse : public Base<Diverse>
{
public:
    Diverse() = default;
    ~Diverse() = default;

    void imp_doit() {std::cout << "fuck you!\n";}
};

int main()
{
    auto&& obj = Diverse();
    obj.doit();
}
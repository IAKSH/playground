#include <iostream>
#include <type_traits>

class Base
{
private:
    float f;

public:
    Base(const float& f) : f(f) {}
    ~Base() = default;

    template <typename T>
    void set(T&& t)
    {
        using U = std::remove_cvref_t<T>;
        if constexpr (std::is_same_v<U,float>)
            f = t;
    }

    template <typename T>
    const T& get()
    {
        using U = std::remove_cvref_t<T>;
        if constexpr (std::is_same_v<U,float>)
            return f;
    }
};

class Diverse : public Base
{
public:
    Diverse(const float& f) : Base(f) {}

    template <typename T>
    void doit(T&& t)
    {
        static_cast<Base&>(*this).doit(t);
        std::cout << "diverse: " << t << std::endl;
    } 
};

int main()
{
    Diverse d;
    d.doit("wdnmd");
    static_cast<Base&>(d).doit("114514");
}
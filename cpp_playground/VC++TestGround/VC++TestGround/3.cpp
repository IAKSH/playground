/*
#include <iostream>
#include <type_traits>

template <typename T, typename... Args> constexpr bool same_type()
{
    if constexpr (sizeof...(Args) == 0)
        return false;
    else
        return (std::same_as<std::remove_reference_t<T>, Args> || ...) || same_type<Args...>();
}

template <typename T>
concept DrawArgs = same_type<T, int, float>();

template <typename T>
struct API
{
    template <DrawArgs U>
    API<T>& operator<< (U&& u)
    {
        using UType = std::remove_cv_t<std::remove_reference_t<U>>;
        if constexpr (std::is_same<UType, int>())
            static_cast<T*>(this)->imp_doInt(u);
        else if constexpr (std::is_same<UType, float>())
            static_cast<T*>(this)->imp_doFloat(u);

        return *this;
    }
};

class MyImp : public API<MyImp>
{
public:
    MyImp() { std::cout << "MyImp()\n"; }
    ~MyImp() { std::cout << "~MyImp()\n"; }
    void imp_doInt(int& i)
    {
        std::cout << "int: " << i << std::endl;
    }
    void imp_doFloat(float& f)
    {
        std::cout << "float: " << f << std::endl;
    }
};

int main()
{
    int a = 114514;
    int& wdnmd = a;
    API<MyImp>&& api = MyImp();
    api << 114 << 5.14f << wdnmd;
}
*/
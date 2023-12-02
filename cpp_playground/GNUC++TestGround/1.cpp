#include <iostream>
#include <type_traits>

template <typename T>
struct API
{
    template <typename U>
    API& operator<< (U&& u)
    {
        using UType = std::remove_cv_t<std::remove_reference_t<U>>;
        if constexpr(std::is_same<UType, int>())
        {
            static_cast<T*>(this)->imp_doInt(u);
            return *this;
        }
        else if constexpr(std::is_same<UType,float>())
        {
            static_cast<T*>(this)->imp_doFloat(u);
            return *this;
        }
        return *this;
    }
};

class MyImp : public API<MyImp>
{
public:
    MyImp(){}
    ~MyImp(){}
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

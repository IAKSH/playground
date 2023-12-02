#include <iostream>
#include <concepts>

template <typename T>
concept AnyInitializerList = requires { typename std::initializer_list<T>; };

template <AnyInitializerList T>
void doit(T t)
{
    for(int i = 0;i < static_cast<int>(t.size());i++)
    {
        std::cout << (*(&*t.begin() + i)).name << std::endl;
    }
}

int main()
{
    struct Person
    {
        int age;
        const char *name;
        const char *getName()
        {
            return name;
        }
    };
    std::initializer_list<Person> list{{24, "tenso"}, {114514, "zun"}};
    doit(list);
}
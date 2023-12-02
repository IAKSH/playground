#include <__chrono/duration.h>
#include <iostream>
#include <chrono>

template <typename T>
struct TypeID
{
    static long long get()
    {
        static long long id = reinterpret_cast<long long>(&id);
        return id;
    }
};

struct A
{
    long long get_rtti_id()
    {
        static long long id = TypeID<A>::get();
        return id;
    }
};

struct B
{
    long long get_rtti_id()
    {
        static long long id = TypeID<B>::get();
        return id;
    }
};

int main()
{
    A a;
    B b;
    
    bool equality {false};
    auto begin = std::chrono::high_resolution_clock::now();
    //for(int i = 0;i < 10000;i++)
    //    equality = a.get_rtti_id() == b.get_rtti_id();
    for(int i = 0;i < 10000;i++)
        equality = typeid(a) == typeid(b);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "us" << std::endl;
    return 0;
}
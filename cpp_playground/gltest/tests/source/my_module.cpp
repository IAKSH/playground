module;
#include <iostream>
#include <format>
export module A; // declares the primary module interface unit for named module 'A'
 
// hello() will be visible by translations units importing 'A'
export char const* hello() { return "hello"; } 
 
// world() will NOT be visible.
char const* world() { return "world"; }
 
// Both one() and zero() will be visible.
export
{
    int one()  { return 1; }
    int zero() { return 0; }
}

class Base {
protected:
    int i;

    virtual int get_num() noexcept {
        return i;
    }

public:
    Base(int i) noexcept 
        : i(i) {
    }

    void doit() noexcept {
        std::cout << "get_num() = " << get_num() << std::endl;
    }
};

class Son : public Base {
private:
    virtual int get_num() noexcept override final {
        return i + 1;
    }

public:
    Son(int i) noexcept 
        : Base(i) {
    }
};
 
// Exporting namespaces also works: hi::english() and hi::french() will be visible.
export namespace hi
{
    char const* english() { return "Hi!"; }
    char const* french()  { return "Salut!"; }

    class Foo {
    public:
        Foo() = default;
        ~Foo() = default;
        void say() {
            for(int i = 0;i < 100;i++) {
                std::cout << std::format("hello! i={}\n",i);
                Base b(114);
                Son s(114);
                b.doit();
                s.doit();
            }
        }
    };
}
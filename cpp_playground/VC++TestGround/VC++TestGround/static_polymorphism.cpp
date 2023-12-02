/*
#include <iostream>

template <typename T>
class Base
{
protected:
	Base() { std::cout << "Base()\n"; }
	~Base() { std::cout << "~Base()\n"; }

public:
	void api(float f){ static_cast<T*>(this)->implementation(f); }
};

class Derived : public Base<Derived>
{
public:
	Derived() { std::cout << "Derived()\n"; }
	~Derived() { std::cout << "~Derived()\n"; }
	void implementation(float f) { std::cout << "Derived::implementation got " << f << std::endl; }
};

int main()
{
	Base<Derived>&& obj = Derived();
	obj.api(114514);
	return 0;
}
*/
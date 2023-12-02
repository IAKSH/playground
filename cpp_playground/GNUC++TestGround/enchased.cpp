#include <iostream>

template <typename T>
class __Enchased
{
private:
	T t;
public:
	__Enchased(const T& t) : t(t) {}
	~__Enchased() = default;
	operator T& () { return t; }
    operator T() { return t; }
};

struct ACorrdX : public __Enchased<float>
{
    ACorrdX(const float& f) : __Enchased(f) {};
};

int main()
{
    ACorrdX a(114.0f);
    std::cout << static_cast<float&>(a) << std::endl;
}
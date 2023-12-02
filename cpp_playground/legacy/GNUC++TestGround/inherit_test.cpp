#include <iostream>

template <typename T>
struct HasX {
    template <typename U>
    static std::true_type test(decltype(&U::x));
    
    template <typename U>
    static std::false_type test(...);
    
    static constexpr bool value = decltype(test<T>(nullptr))::value;
};

template <typename T>
concept WithX = HasX<T>::value;

template <WithX T>
int get_x(const T& t)
{
    return t.x;
}

class Point
{
private:
    int x;

public:
    Point() = default;
    ~Point() = default;

    friend int get_x(const Point& t);
    friend struct HasX<Point>;
};

class Velocitor
{
private:
    int x;

public:
    Velocitor() = default;
    ~Velocitor() = default;

    friend int get_x(const Velocitor& t);
    friend struct HasX<Velocitor>;
};

static_assert(WithX<Point>);
static_assert(WithX<Velocitor>);

int main() noexcept
{
    Point p;
    Velocitor v;
    std::cout << get_x(p) << '\t' << get_x(v) << std::endl;
};
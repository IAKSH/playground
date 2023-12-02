#include <iostream>

template <typename... T>
void print(T&&... t)
{
    ((std::cout << std::forward<T>(t) << '\t'), ...);
}

int main()
{
    print("wo ri ni ma",114.514,1919,810);
}
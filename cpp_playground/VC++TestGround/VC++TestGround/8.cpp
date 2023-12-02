/*
#include <iostream>
#include <type_traits>
#include <concepts>

template <typename T, typename... Args> constexpr bool any_same()
{
	if constexpr (sizeof...(Args) == 0)
		return false;
	else
		return (std::same_as<std::remove_reference_t<T>, Args> || ...) || any_same<Args...>();
}

template <typename T>
concept myConcept = any_same<T, int, float, char>();

template <myConcept T>
void foo(T t)
{
	using U = std::remove_cvref_t<T>;
	if constexpr (std::is_same<U, int>::value)
		std::cout << "int: " << t << std::endl;
	else if constexpr (std::is_same<U, float>::value)
		std::cout << "float: " << t << std::endl;
	else if constexpr (std::is_same<U, char>::value)
		std::cout << "char: " << t << std::endl;
}

int main()
{
	foo(111);
	foo(2.22f);
	foo('a');

	return 0;
}
*/
/*
#include <iostream>
#include <type_traits>
#include <concepts>

enum class HELLO {};
enum class FUCKYOU {};

template<typename T, typename... Args>
constexpr bool same_type() {
	if constexpr (sizeof...(Args) == 0)
		return false;
	else
		return (std::same_as<T, Args> || ...) || same_type<Args...>();
}

template <typename T>
concept MyConcept = same_type<T, HELLO, FUCKYOU>();

template <typename T>
void say() requires MyConcept<T>
{
	if constexpr (std::same_as<T, HELLO>)
	{
		std::cout << "hello\n";
	}
	else if constexpr (std::same_as<T, FUCKYOU>)
	{
		std::cout << "fuck you\n";
	}
}

int main()
{
	say<HELLO>();
	say<FUCKYOU>();
	say<int>();

	//std::cout << has_duplicate_types_c<int, float, FUCKYOU> << std::endl;
}
*/
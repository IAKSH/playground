/*
#include <iostream>
#include <unordered_map>

inline static std::unordered_map<std::string, std::string> map;

void set(std::string_view name, std::string_view val)
{
	map[std::move(std::string(name))] = std::move(std::string(val));
}

std::string_view get(std::string_view name)
{
	return (map[std::move(std::string(name))]);
}

int main()
{
	set("wrnmd", "nmsl");
	std::cout << get("wrnmd") << std::endl;
	std::cout << get("wrnmd") << std::endl;
	std::cout << get("wrnmd") << std::endl;
	return 0;
}
*/
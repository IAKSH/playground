/*
#include <iostream>
#include <memory>

const int& foo(std::unique_ptr<int>& ptr)
{
	int& i = *ptr;
	return i;
}

int main()
{
	auto num = std::make_unique<int>(114514);
	std::cout << foo(num) << std::endl;

	return 0;
}
*/
#include <iostream>
#include <mydll.h>

int main() {
	std::cout << "loaded dll, info: " << get_dll_info() << '\n';
	say("nihao");
	say_hello();
	std::cout << "1.0 + 1.1 = " << add(1.0, 1.1) << '\n';
	system("pause");
	return 0;
}
/*
#include <iostream>
#include <vector>

template <typename T>
class Base
{
public:
	void say(const char* str) { static_cast<T*>(this)->imp_say(str); }
};

class Son : public Base<Son>
{
public:
	Son(){}
	~Son(){}
	void imp_say(const char* str)
	{
		std::cout << str << std::endl;
		std::string a("wdnmd");
		std::string b("wdnmd");
		std::cout << (a == b) << std::endl;
	}

	std::string a(std::string_view str)
	{
		return std::move(std::string(str));
	}
};

int main()
{
	Son s;
	s.say("fuck you!");

	std::cout << s.a("wrnmb") << std::endl;

	return 0;
}
*/

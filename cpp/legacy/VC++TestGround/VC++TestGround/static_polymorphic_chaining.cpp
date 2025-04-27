/*
#include <iostream>

template <typename ConcretePainter>
class Painter
{
protected:
	Painter() { std::cout << "Painter()\n"; }
	~Painter() { std::cout << "~Painter()\n"; }

public:
	template <typename T>
	ConcretePainter& draw(T&& t)
	{
		std::cout << "draw a meta\n";
		return static_cast<ConcretePainter&>(*this);
	}
};

class TexturePainter : public Painter<TexturePainter>
{
public:
	TexturePainter() { std::cout << "TexturePainter()\n"; }
	~TexturePainter() { std::cout << "~TexturePainter()\n"; }

	TexturePainter& bindTexture(int id)
	{
		std::cout << "bind a texture\n";
		return *this;
	}
};

int main()
{
	TexturePainter renderer;
	renderer.bindTexture(114).draw("wdnmd").bindTexture(514).draw("fuck you");
}
*/
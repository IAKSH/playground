/*
#include <iostream>

template <typename T>
class Renderer
{
protected:
	Renderer() { std::cout << "Renderer()\n"; }
	~Renderer() { std::cout << "~Renderer()\n"; }
public:
	Renderer& drawMeta()
	{
		static_cast<T*>(this)->imp_drawMeta();
		return *this;
	}
	Renderer& setColor()
	{
		static_cast<T*>(this)->imp_setColor();
		return *this;
	}
	Renderer& bindTexture()
	{
		static_cast<T*>(this)->imp_bindTexture();
		return *this;
	}
};

class GLRenderer : public Renderer<GLRenderer>
{
public:
	GLRenderer() { std::cout << "GLRenderer()\n"; }
	~GLRenderer() { std::cout << "~GLRenderer()\n"; }
	void imp_drawMeta() { std::cout << "draw a meta\n"; }
	void imp_setColor() { std::cout << "set a color\n"; }
	//void imp_bindTexture() { std::cout << "bind a texture\n"; }
};

int main()
{
	Renderer<GLRenderer>&& ren = GLRenderer();
	ren.setColor().bindTexture().drawMeta();

}
*/
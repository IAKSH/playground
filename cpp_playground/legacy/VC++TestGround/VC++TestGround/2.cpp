/*
#include <iostream>

struct Color
{
	Color() {}
	~Color() = default;
};
struct Texture
{
	Texture() {}
	~Texture() = default;
};
struct Meta
{
	Meta() {}
	~Meta() = default;
};

template<typename T, typename... Args>
constexpr bool same_type()
{
	if constexpr (sizeof...(Args) == 0)
		return false;
	else
		return (std::same_as<T, Args> || ...) || same_type<Args...>();
}

template <typename T>
concept MyConcept = same_type<T, Color, Texture, Meta>();

template <typename T>
class Renderer
{
protected:
	Renderer() { std::cout << "Renderer()\n"; }
	~Renderer() { std::cout << "~Renderer()\n"; }
public:
	template <MyConcept U>
	Renderer& operator<< (U&& t)
	{
		if constexpr (std::same_as<U, Color>)
			static_cast<T*>(this)->imp_setColor();
		else if constexpr (std::same_as<U, Texture>)
			static_cast<T*>(this)->imp_bindTexture();
		else if constexpr (std::same_as<U, Meta>)
			static_cast<T*>(this)->imp_drawMeta();
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
	void imp_bindTexture() { std::cout << "bind a texture\n"; }
};

int main()
{
	Renderer<GLRenderer>&& ren = GLRenderer();
	ren << Color() << Texture() << Meta();
}
*/
#include <iostream>
#include <concepts>

template <typename T>
concept Point2D = requires(T t, float x,float y,float val)
{
	{new T(x,y)} -> std::same_as<T*>;
	{t.get_position_x()} -> std::same_as<float>;
	{t.get_position_y()} -> std::same_as<float>;
	{t.set_position_x(val)} -> std::same_as<void>;
	{t.set_position_y(val)} -> std::same_as<void>;
};

template <typename T>
concept Rectangle = requires(T t,float x,float y,float w,float h,float val)
{
	requires(Point2D<T>);
	{new T(x,y,w,h)} -> std::same_as<T*>;
	{t.get_rect_width()} -> std::same_as<float>;
	{t.get_rect_height()} -> std::same_as<float>;
	{t.set_rect_width(val)} -> std::same_as<void>;
	{t.set_rect_height(val)} -> std::same_as<void>;
};

class MyPoint
{
private:
	float x, y, z;

public:
    MyPoint() :x(0.0f), y(0.0f), z(0.0f) {}
	MyPoint(float x, float y) :x(x), y(y), z(0.0f) {}
	MyPoint(float x, float y, float z) :x(x), y(y), z(z) {}
	~MyPoint() = default;
	float get_position_x() const { return x; }
	float get_position_y() const { return y; }
	float get_position_z() const { return z; }
	void set_position_x(float val) { x = val; }
	void set_position_y(float val) { y = val; }
	void set_position_z(float val) { z = val; }
};

template <Point2D Point>
class MyRect
{
private:
	Point point;
	float width;
	float height;

public:
	MyRect() = default;
	MyRect(float x,float y,float w,float h) :width(w),height(h)
	{
		point.set_position_x(x);
		point.set_position_y(y);
	}

	~MyRect() = default;
	float get_rect_width() const { return width; }
	float get_rect_height() const { return height; }
	void set_rect_width(float val) { width = val; }
	void set_rect_height(float val) { height = val; }
};

int main()
{
	MyRect<MyPoint> rect;

	return 0;
}
#include <iostream>

class HeapObject
{
public:
    HeapObject() {}
    ~HeapObject()
    {
        delete this;
    }
};

class Object : public HeapObject
{
private:
    int num;

public:
    Object() {}
    ~Object() {}
};

int main()
{
    while(true)
    {
        HeapObject&& obj = Object();
    }
}
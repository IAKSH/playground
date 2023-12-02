#include <deque>
#include <iostream>

class Obj
{
private:
    int val;

public:
    Obj(int i)
        : val(i)
    {
    }

    ~Obj()
    {
        std::cout << "obj deleted\n";
    }

    int getVal(){return val;}
};

int main()
{
    std::deque<Obj> objs{Obj(114),Obj(514),Obj(1919),Obj(810)};
    return 0;
}
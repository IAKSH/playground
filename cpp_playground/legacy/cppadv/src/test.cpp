#include <iostream>

#include "src/engine.hpp"

namespace demo
{
    static cppadv::gameplay::Engine engine;;

    class FirstRoom : public cppadv::gameplay::Room<FirstRoom> {
    private:
        cppadv::dtype::GameObject kid;
        cppadv::dtype::GameObject background;

    public:
        FirstRoom() {
            setUID(1);
        }

        ~FirstRoom() = default;

        void imp_onEnter() {
            resrouce.add(cppadv::ResouceManager::Resource::GameObject, 1);// GameObject: kid
            resrouce.add(cppadv::ResouceManager::Resource::GameObject, 2);// GameObject: background
        }

        void imp_onQuit() {

        }
    };
}

int main()
{
    std::cout << "Hello, world!\n";

    cppadv::gameplay::Room<demo::FirstRoom>&& firstRoom = demo::FirstRoom();
    demo::engine.enterRoom( firstRoom);

    return 0;
}

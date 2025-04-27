#pragma once

#include "collision.hpp"
#include "datatype.hpp"
#include "display.hpp"
#include "input.hpp"
#include "log.hpp"
#include "mixer.hpp"
#include "resource.hpp"
#include "time.hpp"

namespace cppadv::gameplay {

    template<typename T>
    struct Room : public dtype::UniqueObject {
        cppadv::ResouceManager resrouce;
        void onQuit() {static_cast<T*>(this)->imp_onQuit();}
        void onEnter() {static_cast<T*>(this)->imp_onEnter();}
    };

    class Engine {
    private:
        cppadv::Renderer objRenderer;
        cppadv::Renderer fontRenderer;
        cppadv::Mixer mixer;
        cppadv::Window window;
        unsigned int roomUID;

    public:
        Engine();
        ~Engine();
        template<typename T>
        void enterRoom(Room<T>& room);
    };
}
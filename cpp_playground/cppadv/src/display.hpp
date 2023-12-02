#pragma once

#include <string>
#include <string_view>

#include "log.hpp"
#include "datatype.hpp"

namespace cppadv {
    class Renderer {
    public:
        Renderer();
        ~Renderer();

        // draw game object
        Renderer& operator<< (dtype::GameObject& obj);
    };

    class Window {
    protected:
        unsigned int getHeight();
        unsigned int getWidth();
        unsigned int getPositionX();
        unsigned int getPositionY();
        unsigned int getAimFPS();
        std::string_view getTitle();
        bool visible;

    public:
        Window();
        ~Window();

        enum class Attrib {
            Height,Width,Title,PositionX,PositionY,AimFPS,Visible
        };

        template <Attrib T>
        struct ReturnType {};

        template <>
        struct ReturnType<Attrib::AimFPS> {
            using type = unsigned int;
        };

        template <>
        struct ReturnType<Attrib::PositionX> {
            using type = unsigned int;
        };

        template <>
        struct ReturnType<Attrib::PositionY> {
            using type = unsigned int;
        };

        template <>
        struct ReturnType<Attrib::Title> {
            using type = std::string_view;
        };

        template <>
        struct ReturnType<Attrib::Width> {
            using type = unsigned int;
        };

        template <>
        struct ReturnType<Attrib::Height> {
            using type = unsigned int;
        };

        template <>
        struct ReturnType<Attrib::Visible> {
            using type = bool;
        };

        template<Attrib T>
        typename ReturnType<T>::type& operator[] (Attrib attrib) {
            switch (attrib) {
            case Attrib::Height:
                return getHeight();
            case Attrib::Width:
                return getWidth();
            case Attrib::Title:
                return getTitle();
            case Attrib::PositionX:
                return getPositionX();
            case Attrib::PositionY:
                return getPositionY();
            case Attrib::AimFPS:
                return getAimFPS();
            case Attrib::Visible:
                return visible;
            default:
                log::writeLog(log::LogLevel::Error, "unknow attrib");
                abort();
            }
        }

        void update();
    };
}
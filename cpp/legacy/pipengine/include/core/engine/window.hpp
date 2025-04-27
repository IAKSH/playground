#pragma once

#include "../concepts/window.hpp"
#include <glad/glad.h>
#include <GLFW/glfw3.h>

namespace pipengine::core
{
    class __GLFW_Window
    {
    private:
        GLFWwindow* win;
        std::string title;

    public:
        __GLFW_Window();
        ~__GLFW_Window();
        std::array<int,2> get_window_position() const;
        int get_window_position_x() const;
        int get_window_position_y() const;
        std::array<int,2> get_window_size() const;
        int get_window_size_w() const;
        int get_window_size_h() const;
        std::string get_window_title() const;

        void set_window_position(std::array<int,2> position);
        void set_window_position_x(int x);
        void set_window_position_y(int y);
        void set_window_size(std::array<int,2> size);
        void set_window_size_w(int w);
        void set_window_size_h(int h);
        void set_window_title(std::string_view title);
    };

    static_assert(concepts::Window<__GLFW_Window>);
    using Window = __GLFW_Window;
};
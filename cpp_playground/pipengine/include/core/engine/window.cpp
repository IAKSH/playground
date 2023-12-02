#include "window.hpp"

pipengine::core::__GLFW_Window::__GLFW_Window()
{

}

pipengine::core::__GLFW_Window::~__GLFW_Window()
{
    glfwDestroyWindow(win);
}
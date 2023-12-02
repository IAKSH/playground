#pragma once

#include <array>
#include <string>
#include <concepts>

namespace pipengine::concepts
{
    template <typename T>
    concept Window = requires(T t,int x,int y,int w,int h,std::array<int,2> position,std::array<int,2> size,std::string_view title)
    {
        {t.get_window_position()} -> std::same_as<std::array<int,2>>;
        {t.get_window_position_x()} -> std::same_as<int>;
        {t.get_window_position_y()} -> std::same_as<int>;
        {t.get_window_size()} -> std::same_as<std::array<int,2>>;
        {t.get_window_size_w()} -> std::same_as<int>;
        {t.get_window_size_h()} -> std::same_as<int>;
        {t.get_window_title()} -> std::same_as<std::string>;

        {t.set_window_position(position)} -> std::same_as<void>;
        {t.set_window_position_x(x)} -> std::same_as<void>;
        {t.set_window_position_y(y)} -> std::same_as<void>;
        {t.set_window_size(size)} -> std::same_as<void>;
        {t.set_window_size_w(w)} -> std::same_as<void>;
        {t.set_window_size_h(h)} -> std::same_as<void>;
        {t.set_window_title(title)} -> std::same_as<void>;
    };
}
#pragma once

#include <array>
#include <concepts>

namespace pipengine::concepts
{
    template <typename T>
    concept Pointable = requires(T t,float x,float y,float z)
    {
        {t.get_position_x()} -> std::same_as<float>;
        {t.get_position_y()} -> std::same_as<float>;
        {t.get_position_z()} -> std::same_as<float>;
        
        {t.set_position_x(x)} -> std::same_as<void>;
        {t.set_position_y(y)} -> std::same_as<void>;
        {t.set_position_z(z)} -> std::same_as<void>;
    };

    template <typename T>
    concept Velociable = requires(T t,float vx,float vy,float vz)
    {
        {t.get_velocity_x()} -> std::same_as<float>;
        {t.get_velocity_y()} -> std::same_as<float>;
        {t.get_velocity_z()} -> std::same_as<float>;
        
        {t.set_velocity_x(vx)} -> std::same_as<void>;
        {t.set_velocity_y(vy)} -> std::same_as<void>;
        {t.set_velocity_z(vz)} -> std::same_as<void>;
    };

    template <typename T>
    concept Rotatable = requires(T t,float front,float right,float up)
    {
        {t.get_orientation_quat()} -> std::same_as<std::array<float,4>>;
        {t.rotate(front,right,up)} -> std::same_as<void>;
        {t.move(front,right,up)} -> std::same_as<void>;
    };

    template <typename T>
    concept WithWidthAndHeight = requires(T t,float w,float h)
    {
        {t.get_width()} -> std::same_as<float>;
        {t.get_height()} -> std::same_as<float>;
        
        {t.set_width(w)} -> std::same_as<void>;
        {t.set_height(h)} -> std::same_as<void>;
    };

    template <typename T>
    concept WithRadius = requires(T t,float radius)
    {
        {t.get_radius()} -> std::same_as<float>;
        
        {t.set_radius(radius)} -> std::same_as<void>;
    };

    template <typename T>
    concept Square = Pointable<T> && Velociable<T> && Rotatable<T> && WithWidthAndHeight<T>;

    template <typename T>
    concept Ball = Pointable<T> && Velociable<T> && Rotatable<T> && WithRadius<T>;

    template <typename T>
    concept PhysicsModel = Square<T> || Ball<T>;
};
#pragma once

#include <array>

namespace jumping_ball::misc {
    class Point {
    public:
        float& x{xyz[0]};
        float& y{xyz[1]};
        float& z{xyz[2]};

        Point() noexcept;
        Point(const float& x,const float& y,const float& z) noexcept;

    private:
        std::array<float,3> xyz;
    };

    class RotatablePoint : public Point {
    public:
        std::array<float,4> orientation;
        std::array<float,3> right;
        std::array<float,3> up;
        float getYaw() const noexcept;
        float getPitch() const noexcept;
        float getRoll() const noexcept;

        RotatablePoint() noexcept;
        ~RotatablePoint() = default;
        void rotate(float dUp,float dRight,float dRoll) noexcept;
        void move(float dFront,float dRight,float dHeight) noexcept;

    private:
        void updateVectors() noexcept;
    };
}
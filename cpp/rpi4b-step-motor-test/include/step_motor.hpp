#pragma once
#include <array>

namespace yschw {
    class StepMotor {
    public:
        StepMotor(int in1,int in2,int in3,int in4) noexcept;
        ~StepMotor() = default;
        void step(unsigned int steps) noexcept;
        void angle(float angle) noexcept;
    private:
        std::array<int,4> pins;
        unsigned int angle_to_step(float angle) noexcept;
        void pin_out(const std::array<bool,4>& states) noexcept;
    };
}
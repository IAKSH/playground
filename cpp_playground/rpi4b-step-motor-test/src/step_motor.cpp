#include <step_motor.hpp>
#include <wiringPi.h>

unsigned int yschw::StepMotor::angle_to_step(float angle) noexcept {
    const float step_angle = 0.0879f; // 每步对应的角度
    return static_cast<unsigned int>(angle / step_angle + 0.5f); 
}

void yschw::StepMotor::pin_out(const std::array<bool,4>& states) noexcept {
    for(int i = 0;i < 4;i++)
        digitalWrite(pins[i], states[i]);
    delay(2);
}

yschw::StepMotor::StepMotor(int in1,int in2,int in3,int in4) noexcept {
    pinMode(in1,OUTPUT);
    pinMode(in2,OUTPUT);
    pinMode(in3,OUTPUT);
    pinMode(in4,OUTPUT);
    pins[0] = in1;
    pins[1] = in2;
    pins[2] = in3;
    pins[3] = in4;
}

void yschw::StepMotor::step(unsigned int steps) noexcept {
    constexpr std::array<std::array<bool,4>,8> STEP_SEQUENCE {
        std::array<bool,4>{1,0,0,0},
        std::array<bool,4>{1,1,0,0},
        std::array<bool,4>{0,1,0,0},
        std::array<bool,4>{0,1,1,0},
        std::array<bool,4>{0,0,1,0},
        std::array<bool,4>{0,0,1,1},
        std::array<bool,4>{0,0,0,1},
        std::array<bool,4>{1,0,0,1}
    };
    for(int i = 0; i < steps; i++)
        pin_out(STEP_SEQUENCE[i % STEP_SEQUENCE.size()]);
}

void yschw::StepMotor::angle(float angle) noexcept {
    step(angle_to_step(angle));
}
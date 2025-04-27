#include <wiringPi.h>
#include <iostream>
#include <step_motor.hpp>

#define IN1 0  // GPIO17
#define IN2 3  // GPIO22
#define IN3 2  // GPIO27
#define IN4 1  // GPIO18

void init_beep() noexcept {
    pinMode(29,OUTPUT);
}

void beep() noexcept {
    for(int i = 0;i < 1;i++) {
        digitalWrite(29,1);
        delay(5);
        digitalWrite(29,0);
        delay(5);
    }
}

int main() {
    if (wiringPiSetup() == -1) {
        std::cerr << "failed to init WiringPi\n";
        return 1;
    }

    init_beep();

    yschw::StepMotor motor(IN1,IN2,IN3,IN4);

    beep();
    //motor.step(1024);    
    motor.angle(180.0f);
    beep();

    return 0;
}
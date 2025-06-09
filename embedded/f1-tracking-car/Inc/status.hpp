#pragma once
#include <cstdint>
#include "ultra_sonic.hpp"
#include "ticker.hpp"

enum class LedStatus {
    OFF,KEEP,BEEP,RGB
};

enum class DriveStatus {
    STOP,AUTO,MANUAL,IR_TEST
};

enum class ManualDriveCommand {
    FORWARD,BACKWORD,TURN_LEFT,TURN_RIGHT,STOP
};

extern car::Ticker ticker;
extern car::UlatraSonic ultra_sonic;
extern float distance;

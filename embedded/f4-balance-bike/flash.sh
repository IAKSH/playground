#!/bin/bash

# Flash the program to the target device
OPENOCD=/usr/bin/openocd
RELEASE_DIRECTORY=$(dirname "${BASH_SOURCE[0]}")/build
TARGET=BalanceBike

"$OPENOCD" -f ./openocd.cfg -c "program $RELEASE_DIRECTORY/$TARGET.elf verify reset exit"

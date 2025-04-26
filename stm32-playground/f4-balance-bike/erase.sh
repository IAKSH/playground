#!/bin/bash

# Erase the target device
OPENOCD=/usr/bin/openocd

"$OPENOCD" -f ./openocd.cfg -c "init; reset halt; stm32f4x mass_erase 0; exit"

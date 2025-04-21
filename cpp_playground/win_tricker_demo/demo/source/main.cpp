#include <iostream>
#include <trick/trick.hpp>

int main() {
    trick::ScreenRecorder&& recorder = trick::ScreenRecorderImpl(2560,1440);
    auto bmp = recorder.capture();
    bmp->save_to_file("out.bmp");
    return 0;
}
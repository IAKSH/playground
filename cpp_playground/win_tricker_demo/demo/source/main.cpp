#include <thread>
#include <trick/trick.hpp>
#include <windows.h>

int main() {
    trick::ScreenRecorder&& recorder = trick::ScreenRecorderImpl();
    auto bmp = recorder.capture();
    bmp->save_to_file("out.bmp");

    trick::ScreenBlocker&& blocker = trick::ScreenBlockerImpl(*bmp);
    trick::Beeper&& beeper = trick::BeeperImpl();
    
    beeper.start_random_beep(50,100);
    blocker.show();
    std::this_thread::sleep_for(std::chrono::seconds(10));
    blocker.hide(); 
    beeper.stop_random_beep();

    //while(true)
    //    std::this_thread::sleep_for(std::chrono::seconds(1));

    return 0;
}
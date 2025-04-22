#include <thread>
#include <trick/trick.hpp>

int main() {
    trick::ScreenRecorder&& recorder = trick::ScreenRecorderImpl();
    auto bmp = recorder.capture();
    bmp->save_to_file("out.bmp");

    trick::ScreenBlocker&& blocker = trick::ScreenBlockerImpl(*bmp);

    blocker.show();
    std::this_thread::sleep_for(std::chrono::seconds(10));
    blocker.hide(); 

    return 0;
}
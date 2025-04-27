#include <trick/trick_c.h>
#include <trick/trick.hpp>
#include <memory>
#include <string>
#include <new>

// 这里在 C++ 实现文件中定义不透明结构体的内部内容
struct trick_Bitmap {
    // 采用 shared_ptr 保证捕获到的 Bitmap 能够在多个地方共享管理
    std::shared_ptr<trick::Bitmap> instance;
};

struct trick_ScreenRecorder {
    // 我们采用 unique_ptr 管理屏幕录制器实例
    std::unique_ptr<trick::ScreenRecorderImpl> instance;
};

extern "C" {
trick_ScreenRecorder* trick_screen_recorder_create(int width, int height) {
    try {
        auto recorder = new trick_ScreenRecorder;
        recorder->instance = std::make_unique<trick::ScreenRecorderImpl>(width, height);
        return recorder;
    }
    catch (...) {
        return nullptr;
    }
}

void trick_screen_recorder_destroy(trick_ScreenRecorder* recorder) {
    if (recorder) {
        delete recorder;
    }
}

trick_Bitmap* trick_screen_recorder_capture(trick_ScreenRecorder* recorder) {
    if (!recorder || !recorder->instance) return nullptr;
    try {
        auto bmp_shared = recorder->instance->capture();
        auto bmp_wrapper = new trick_Bitmap;
        bmp_wrapper->instance = bmp_shared;
        return bmp_wrapper;
    }
    catch (...) {
        return nullptr;
    }
}

int trick_bitmap_width(const trick_Bitmap* bmp) {
    return (bmp && bmp->instance) ? bmp->instance->width() : 0;
}

int trick_bitmap_height(const trick_Bitmap* bmp) {
    return (bmp && bmp->instance) ? bmp->instance->height() : 0;
}

void trick_bitmap_save_to_file(const trick_Bitmap* bmp, const char* path) {
    if (bmp && bmp->instance && path) {
        // 将 C 字符串转换为 std::string_view 传入 C++ 接口
        bmp->instance->save_to_file(std::string_view(path));
    }
}

void trick_bitmap_destroy(trick_Bitmap* bmp) {
    delete bmp;
}
} // extern "C"

#pragma once
#include <string_view>
#include <memory>
#include <concepts>

namespace trick::base {
template <typename T>
concept BitmapImpl = requires(const T bmp, std::string_view path) {
    { bmp.width_impl() } -> std::convertible_to<int>;
    { bmp.height_impl() } -> std::convertible_to<int>;
    { bmp.data_impl() } -> std::convertible_to<const char*>;
    { bmp.save_to_file_impl(path) } -> std::same_as<void>;
} &&
    !std::default_initializable<T> &&
    std::constructible_from<T, T&>;

template <typename Derived>
struct Bitmap {
    int width() const {
        return static_cast<const Derived*>(this)->width_impl();
    }

    int height() const {
        return static_cast<const Derived*>(this)->height_impl();
    }

    char* data() const {
        //return const_cast<char*>();
        return static_cast<const Derived*>(this)->data_impl();
    }

    void save_to_file(std::string_view path) const {
        static_cast<const Derived*>(this)->save_to_file_impl(path);
    }
};

template <typename T, typename BitmapImplT>
concept ScreenRecorderImpl = requires(T recorder) {
    { recorder.capture_impl() } -> std::convertible_to<std::shared_ptr<Bitmap<BitmapImplT>>>;
} && 
    std::default_initializable<T> &&
    std::constructible_from<T,T&> &&
    std::constructible_from<T,int,int> && 
    BitmapImpl<BitmapImplT>;

template <typename Derived,typename BitmapDerived>
struct ScreenRecorder {
    std::shared_ptr<Bitmap<BitmapDerived>> capture() {
        return static_cast<Derived*>(this)->capture_impl();
    }
};

template <typename T, typename BitmapImplT>
concept ScreenBlockerImpl = requires(T blocker) {
    { blocker.show_impl() } -> std::same_as<void>;
    { blocker.hide_impl() } -> std::same_as<void>;
} &&
    !std::default_initializable<T> &&
    !std::constructible_from<T,T&> &&
    std::constructible_from<T,const Bitmap<BitmapImplT>&> &&
    BitmapImpl<BitmapImplT>;

template <typename Derived>
struct ScreenBlocker {
    void show() {
        static_cast<Derived*>(this)->show_impl();
    }

    void hide() {
        static_cast<Derived*>(this)->hide_impl();
    }
};

template <typename T>
concept BeeperImpl = requires(T beeper,int ms,int min_ms,int max_ms,int freq) {
    { beeper.beep_impl(ms) } -> std::same_as<void>;
    { beeper.start_random_beep_impl(min_ms,max_ms) } -> std::same_as<void>;
    { beeper.stop_random_beep_impl() } -> std::same_as<void>;
    { beeper.set_freq_impl(freq) } -> std::same_as<void>;
} && 
    std::default_initializable<T> &&
    !std::constructible_from<T,T&> &&
    std::constructible_from<T,int>;

template <typename Derived>
struct Beeper {
    void beep(int ms) const {
        static_cast<const Derived*>(this)->beep_impl(ms);
    }

    void start_random_beep(int min_ms,int max_ms) {
        static_cast<Derived*>(this)->start_random_beep_impl(min_ms,max_ms);
    }

    void stop_random_beep() {
        static_cast<Derived*>(this)->stop_random_beep_impl();
    }

    void set_freq(int freq) {
        static_cast<Derived*>(this)->set_freq_impl();   
    }
};

template <typename T>
concept BurnerImpl = requires(T burner,int n) {
    { burner.run_impl() } -> std::same_as<void>;
    { burner.stop_impl() } -> std::same_as<void>;
    { burner.set_worker_num_impl(n) } -> std::same_as<void>;
} &&
    std::default_initializable<T> &&
    !std::constructible_from<T,T&> &&
    std::constructible_from<T,int>;

template <typename Derived>
struct Burner {
    void run() {
        static_cast<Derived*>(this)->run_impl();
    }

    void stop() {
        static_cast<Derived*>(this)->stop_impl();
    }

    void set_worker_num(int n) {
        static_cast<Derived*>(this)->set_worker_num_impl();
    }
};
}
#pragma once
#include <string_view>
#include <memory>
#include <concepts>

namespace trick::base {
template <typename T>
concept BitmapImplConcept = requires(const T bmp, std::string_view path) {
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

template <typename T, typename BitmapImpl>
concept ScreenRecorderImplConcept = requires(T recorder) {
    { recorder.capture_impl() } -> std::convertible_to<std::shared_ptr<Bitmap<BitmapImpl>>>;
} && 
    std::default_initializable<T> &&
    std::constructible_from<T,T&> &&
    std::constructible_from<T,int,int> && 
    BitmapImplConcept<BitmapImpl>;

template <typename Derived,typename BitmapDerived>
struct ScreenRecorder {
    std::shared_ptr<Bitmap<BitmapDerived>> capture() {
        return static_cast<Derived*>(this)->capture_impl();
    }
};

template <typename T, typename BitmapImpl>
concept ScreenBlockerImpl = requires(const T blocker) {
    { blocker.show_impl() } -> std::same_as<void>;
    { blocker.hide_impl() } -> std::same_as<void>;
} &&
    !std::default_initializable<T> &&
    !std::constructible_from<T,T&> &&
    std::constructible_from<T,const Bitmap<BitmapImpl>&> &&
    BitmapImplConcept<BitmapImpl>;

template <typename Derived>
struct ScreenBlocker {
    void show() {
        static_cast<const Derived*>(this)->show_impl();
    }

    void hide() {
        static_cast<const Derived*>(this)->hide_impl();
    }
};
}
#pragma once
#include <trick/trick_base.hpp>

#if defined(_WIN32)
#include <trick/trick_impl_win.hpp>
#elif defined(__linux__)
#error "linux impl is WIP"
#else
#error "unsupported platform"
#endif

namespace trick {
#if defined(_WIN32)
    using BitmapImpl = win::Bitmap;
    using ScreenRecorderImpl = win::ScreenRecorder;
    using ScreenBlockerImpl = win::ScreenBlocker;
#endif

using Bitmap = base::Bitmap<BitmapImpl>;
using ScreenRecorder = base::ScreenRecorder<ScreenRecorderImpl,BitmapImpl>;
using ScreenBlocker = base::ScreenBlocker<ScreenBlockerImpl>;
}
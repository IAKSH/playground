#include <spdlog/spdlog.h>

int main() noexcept {
    spdlog::set_level(spdlog::level::info);
    
    spdlog::trace("this is a {}!","trace");
    spdlog::debug("this is a {}!","debug");
    spdlog::info("Hello, {}!", "World");
    spdlog::warn("this is a {}!","warn");

    spdlog::set_pattern("%^[%H:%M:%S] [thread %t] [%l] %v%$");
    // https://github.com/gabime/spdlog/wiki/3.-Custom-formatting

    spdlog::error("this is a {}!","error");
    spdlog::critical("this is a {}!","critical");

    spdlog::set_level(spdlog::level::trace);

    SPDLOG_TRACE("nihao {}?","ma");
    SPDLOG_DEBUG("ni{}?","hao");
    SPDLOG_INFO("???");

    return 0;
}
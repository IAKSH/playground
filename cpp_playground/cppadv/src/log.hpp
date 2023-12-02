#pragma once

#include <ostream>
#include <string_view>

namespace cppadv::log {
    enum class LogLevel {
        Log,Warning,Error
    };

    inline static std::ostream stream();
    void initialize();
    void writeLog(LogLevel level,std::string_view str);
}
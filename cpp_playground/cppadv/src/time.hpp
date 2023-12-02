#pragma once

#include "datatype.hpp"

namespace cppadv::time {
    void initialize();
    dtype::TimePoint getTime();
    dtype::Millisecond getTimeDiffer(const dtype::TimePoint& t1,const dtype::TimePoint& t2);
}
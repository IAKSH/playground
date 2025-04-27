#pragma once

#include "datatype.hpp"

namespace cppadv::col {
    bool collisionCheckGJK(const dtype::GameObject& obj1,const dtype::GameObject& obj2);
    bool collisionCheckAABB(const dtype::GameObject& obj1,const dtype::GameObject& obj2);
}
#pragma once

#include "../concepts/event.hpp"

namespace pipengine::core
{
    class BaseEvent
    {
    protected:
        BaseEvent(concepts::CoreEventTypeMark typemark) : typemark(typemark) {}
        concepts::CoreEventTypeMark typemark;

    public:
        virtual ~BaseEvent() = default;
        concepts::CoreEventTypeMark get_event_typemark() {return typemark;}
    };

    static_assert(concepts::WithEventTypeMark<BaseEvent>);
};
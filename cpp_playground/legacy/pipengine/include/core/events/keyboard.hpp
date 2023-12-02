#pragma once

#include "base.hpp"

namespace pipengine::core
{
    class KeyButtomEvent : BaseEvent
    {
    private:
        concepts::KeyButtonCode code;
        concepts::KeyButtonStatus status;

    public:
        KeyButtomEvent(concepts::KeyButtonCode code)
            : BaseEvent(concepts::CoreEventTypeMark::KeyButton),code(code)
        {
        }

        virtual ~KeyButtomEvent() override = default;

        auto get_key_button_code() const {return code;}
        auto get_key_button_status() const {return status;}
    };

    static_assert(concepts::WithKeyButtonCode<KeyButtomEvent> && concepts::WithKeyStatus<KeyButtomEvent>);
}
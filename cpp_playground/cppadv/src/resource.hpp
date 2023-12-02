#pragma once

#include "datatype.hpp"
#include <deque>
#include <functional>

namespace cppadv {
    class ResouceManager {
    private:
        std::deque<cppadv::dtype::Image> images;
        std::deque<cppadv::dtype::Audio> audios;
        std::deque<cppadv::dtype::Animation> animations;
        std::deque<cppadv::dtype::GameObject> gameobjects;

    public:
        ResouceManager();
        ~ResouceManager();

        template<typename T>
        typename ReturnType<T>::type& operator[] (Resource res) {
            switch (res) {
                case Resource::Image:
                    return [&](unsigned int uid) {
                        for(auto& item : images) {
                            if(item.getUID() == uid)
                                return item;
                        }
                        log::writeLog(log::LogLevel::Error, "can't find image");
                        abort();
                    };
                case Resource::Audio:
                    return [&](unsigned int uid) {
                        for(auto& item : audios) {
                            if(item.getUID() == uid)
                                return item;
                        }
                        log::writeLog(log::LogLevel::Error, "can't find audio");
                        abort();
                    };
                case Resource::Animation:
                    return [&](unsigned int uid) {
                        for(auto& item : animations) {
                            if(item.getUID() == uid)
                                return item;
                        }
                        log::writeLog(log::LogLevel::Error, "can't find animation");
                        abort();
                    };
                case Resource::GameObject:
                    return [&](unsigned int uid) {
                        for(auto& item : gameobjects) {
                            if(item.getUID() == uid)
                                return item;
                        }
                        log::writeLog(log::LogLevel::Error, "can't find gameobject");
                        abort();
                    };
                default:
                    log::writeLog(log::LogLevel::Error, "unknow resrouce type enum");
                    abort();
            }

        }

        void remove(Resource res,unsigned int uid);
        void add(Resource res,unsigned int uid);
    };
}
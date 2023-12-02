#pragma once

#include <array>
#include <chrono>
#include <deque>
#include <initializer_list>
#include <iostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>

#include "log.hpp"

namespace cppadv::dtype {

    using PixelCoord = unsigned int;
    using Millisecond = unsigned int;
    using TimePoint = std::chrono::steady_clock::time_point;

    class UniqueObject {
    private:
        unsigned int uid;

    public:
        unsigned int getUID() {return uid;}
        void setUID(unsigned int i) {uid = i};
    };

    class Image : public UniqueObject {
    public:
        Image(unsigned int id);
        ~Image();
    };

    class Audio : public UniqueObject {
    public:
        Audio(unsigned int id);
        ~Audio();
    };

    class Animation : public UniqueObject {
    private:
        class AnimationSet {
        private:
            std::deque<Image> images;
            Millisecond interval;

        public:
            AnimationSet();
            ~AnimationSet();
            void addImage(const Image& img) const;
            void setInterval(Millisecond ms);
            Image& getImageAt(int i);
        };

        std::unordered_map<std::string,AnimationSet> imageSets;
        AnimationSet* currentAnimation;
        int currentIndex;
        TimePoint lastUpdate;

    public:
        Animation();
        ~Animation();
        void addImageToAnimation(std::string_view name,std::initializer_list<Image> imgs);
        void switchAnimation(std::string_view name);
        void tryUpdateImage();
        void getCurrentImage();
    };

    class GameObject : public UniqueObject {
    private:
        unsigned int x,y,z;
        unsigned int vx,vy,vz;
        unsigned int rx,ry,rz;
        unsigned char red,green,blue,alpha; 
        float gain,pitch;
        Audio* playingAudio;
        Animation* animation;
        std::array<std::array<float,2>,4> texCoords;
    
    public:
         GameObject();
        ~GameObject();

        enum class Attrib {
            PositionX,PositionY,PositionZ,
            VelocityX,VelocityY,VelocityZ,
            RotateX,RotateY,RotateZ,
            Gain,Pitch,
            TexCoordX0,TexCoordX1,TexCoordX2,TexCoordX3,
            TexCoordY0,TexCoordY1,TexCoordY2,TexCoordY3,
            RED,GREEN,BLUE,ALPHA
        };
        
        template <Attrib T>
        struct ReturnType {};

        template <>
        struct ReturnType<Attrib::PositionX> {
            using type = unsigned int;
        };

        template <>
        struct ReturnType<Attrib::PositionY> {
            using type = unsigned int;
        };

        template <>
        struct ReturnType<Attrib::PositionZ> {
            using type = unsigned int;
        };

        template <>
        struct ReturnType<Attrib::VelocityX> {
            using type = unsigned int;
        };

        template <>
        struct ReturnType<Attrib::VelocityY> {
            using type = unsigned int;
        };

        template <>
        struct ReturnType<Attrib::VelocityZ> {
            using type = unsigned int;
        };

        template <>
        struct ReturnType<Attrib::RotateX> {
            using type = unsigned int;
        };

        template <>
        struct ReturnType<Attrib::RotateY> {
            using type = unsigned int;
        };

        template <>
        struct ReturnType<Attrib::RotateZ> {
            using type = unsigned int;
        };

        template <>
        struct ReturnType<Attrib::Pitch> {
            using type = float;
        };

        template <>
        struct ReturnType<Attrib::Gain> {
            using type = float;
        };

        template <>
        struct ReturnType<Attrib::TexCoordX0> {
            using type = float;
        };

        template <>
        struct ReturnType<Attrib::TexCoordX1> {
            using type = float;
        };

        template <>
        struct ReturnType<Attrib::TexCoordX2> {
            using type = float;
        };

        template <>
        struct ReturnType<Attrib::TexCoordX3> {
            using type = float;
        };

        template <>
        struct ReturnType<Attrib::TexCoordY0> {
            using type = float;
        };

        template <>
        struct ReturnType<Attrib::TexCoordY1> {
            using type = float;
        };

        template <>
        struct ReturnType<Attrib::TexCoordY2> {
            using type = float;
        };

        template <>
        struct ReturnType<Attrib::TexCoordY3> {
            using type = float;
        };

        template <>
        struct ReturnType<Attrib::RED> {
            using type = unsigned char;
        };

        template <>
        struct ReturnType<Attrib::GREEN> {
            using type = unsigned char;
        };

        template <>
        struct ReturnType<Attrib::BLUE> {
            using type = unsigned char;
        };

        template <>
        struct ReturnType<Attrib::ALPHA> {
            using type = unsigned char;
        };

        // set/get attrib
        // eg: myRect[Rectangle::Attrib::PositionX] = 10;
        template<Attrib T>
        typename ReturnType<T>::type& operator[] (Attrib attrib) {
            switch (attrib) {
            case Attrib::PositionX:
                return x;
            case Attrib::PositionY:
                return y;
            case Attrib::PositionZ:
                return z;
            case Attrib::VelocityX:
                return vx;
            case Attrib::VelocityY:
                return vy;
            case Attrib::VelocityZ:
                return vz;
            case Attrib::RotateX:
                return rx;
            case Attrib::RotateY:
                return ry;
            case Attrib::RotateZ:
                return rz;
            case Attrib::Pitch:
                return pitch;
            case Attrib::Gain:
                return gain;
            case Attrib::TexCoordX0:
                return texCoords[0][0];
            case Attrib::TexCoordX1:
                return texCoords[0][1];
            case Attrib::TexCoordX2:
                return texCoords[0][2];
            case Attrib::TexCoordX3:
                return texCoords[0][3];
            case Attrib::TexCoordY0:
                return texCoords[1][0];
            case Attrib::TexCoordY1:
                return texCoords[1][1];
            case Attrib::TexCoordY2:
                return texCoords[1][2];
            case Attrib::TexCoordY3:
                return texCoords[1][3];
            case Attrib::RED:
                return red;
            case Attrib::GREEN:
                return green;
            case Attrib::BLUE:
                return blue;
            case Attrib::ALPHA:
                return alpha;
            default:
                log::writeLog(log::LogLevel::Error, "unknow attrib");
                abort();
            }
        }

        void playAudio(const Audio& audio) const;
        void pauseAudio();
        void resumeAudio();
        void rewindAudio();
        Animation& getAnimation();
    };
}
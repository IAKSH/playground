#pragma once

#include <jumping_ball/graphics.hpp>
#include <jumping_ball/physics.hpp>
#include <jumping_ball/audio.hpp>

namespace jumping_ball::gameobject {

    using namespace graphics;
    using namespace audio;
    using namespace physics;

    class GameObject {
    public:
        std::shared_ptr<RenPipe> ren_pipe;
        std::unique_ptr<RigidBody> rigid_body;
        AudioPipe audio_pipe;

        GameObject(std::shared_ptr<RenPipe> ren_pipe,std::unique_ptr<RigidBody> rigid_body) noexcept
            : ren_pipe(ren_pipe), rigid_body(std::move(rigid_body))
        {

        }

        ~GameObject() noexcept {

        }

        void update(float delta_time) noexcept {
            rigid_body->update(delta_time);
            audio_pipe.setPosition(rigid_body->position);
            audio_pipe.setVelocity(rigid_body->velocity);
        }
        
        // 由外部决定如何绘制
        //void draw() noexcept {
        //
        //}
    };
}
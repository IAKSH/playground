#pragma once

#include <jumping_ball/graphics.hpp>
#include <jumping_ball/physics.hpp>
#include <jumping_ball/audio.hpp>
#include <glm/gtc/quaternion.hpp>
#include <functional>

namespace jumping_ball::gameobject {

    using namespace graphics;
    using namespace audio;
    using namespace physics;

    template <typename T>
    class CollisionCallback : public btCollisionWorld::ContactResultCallback {
    public:
        CollisionCallback(T& t, std::function<void(T& t)> callback) : t(t), callback(callback) {}

        virtual btScalar addSingleResult(btManifoldPoint& cp,
            const btCollisionObjectWrapper* colObj0Wrap, int partId0, int index0,
            const btCollisionObjectWrapper* colObj1Wrap, int partId1, int index1) override {
            callback(t);
            return 0; // return value not used in Bullet 2.8x
        }

    private:
        T& t;
        std::function<void(T& t)> callback;
    };

    class GameObject {
    public:
        std::shared_ptr<RenPipe> ren_pipe;// 要不直接删掉
        AudioPipe audio_pipe;
        btRigidBody* body;

        GameObject(std::shared_ptr<RenPipe> ren_pipe) noexcept;
        ~GameObject() noexcept;
        glm::vec3 getPosition() noexcept;
        glm::vec3 getVelocity() noexcept;
        glm::quat getRotate() noexcept;
        void applyForce(const glm::vec3& target_position, float force_magnitude = 1.0f) noexcept;
        void setCollisionCallback(std::function<void(GameObject&)> callback) noexcept;
        void checkCollision() noexcept;

    private:
        std::unique_ptr<CollisionCallback<GameObject>> collision_callback;
    };
}
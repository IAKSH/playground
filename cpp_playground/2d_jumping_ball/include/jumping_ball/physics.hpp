#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <memory>
#include <vector>

namespace jumping_ball::physics {
    class RigidBody {
    public:
        glm::vec3 position;                             // 位置
        glm::vec3 velocity;                             // 速度
        glm::vec3 angular_velocity;                     // 角速度
        float mass;                                     // 质量
        glm::mat3 inertia_tensor;                       // 惯性张量
        glm::vec3 force;                                // 力
        glm::vec3 torque;                               // 扭矩
        glm::quat orientation;                          // 方向
        std::shared_ptr<std::vector<float>> vertices;   // 刚体的顶点
        glm::vec3 centerOfMass;                         // 重心

        RigidBody() noexcept;
        RigidBody(std::shared_ptr<std::vector<float>> vertices) noexcept;
        ~RigidBody() = default;

        void addVertex(float x, float y, float z) noexcept;
        void updateCenterOfMass() noexcept;
        void update(float deltaTime) noexcept;
        void applyForce(const glm::vec3& f, const glm::vec3& point) noexcept;

    private:
        void applyDefaultValue() noexcept;
    };

    //bool checkCollision(RigidBody& rb1, RigidBody& rb2) noexcept {
    //    // TODO
    //    return false;
    //}
}
#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>

namespace jumping_ball::physics {
    class RigidBody {
    public:
        glm::vec3 position;                 // 位置
        glm::vec3 velocity;                 // 速度
        glm::vec3 angular_velocity;         // 角速度
        float mass;                         // 质量
        glm::mat3 inertia_tensor;           // 惯性张量
        glm::vec3 force;                    // 力
        glm::vec3 torque;                   // 扭矩
        glm::quat orientation;              // 方向
        std::vector<glm::vec3> vertices;    // 刚体的顶点
        glm::vec3 centerOfMass;             // 重心

        RigidBody();

        void addVertex(const glm::vec3& vertex);
        void updateCenterOfMass();
        void update(float deltaTime);
        void applyForce(const glm::vec3& f, const glm::vec3& point);
    };

    //bool checkCollision(RigidBody& rb1, RigidBody& rb2) noexcept {
    //    // TODO
    //    return false;
    //}

    struct Ball : public RigidBody {
        const float radius = 50.0f;

        Ball() noexcept {
            mass = 0.5f;

            constexpr int segments = 360;
            for (int i = 0; i < segments; ++i) {
                float theta = 2.0f * 3.1415926f * float(i) / float(segments);
                float x = cosf(theta);
                float y = sinf(theta);
                addVertex(glm::vec3(x, y, 0.0f));
            }
        }
    };
}
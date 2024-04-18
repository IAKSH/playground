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

        RigidBody() {
            position = glm::vec3(0.0f);
            velocity = glm::vec3(0.0f);
            angular_velocity = glm::vec3(0.0f);
            mass = 1.0f;
            inertia_tensor = glm::mat3(1.0f);
            force = glm::vec3(0.0f);
            torque = glm::vec3(0.0f);
            orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
            centerOfMass = glm::vec3(0.0f);
        }

        // 添加顶点的函数
        void addVertex(const glm::vec3& vertex) {
            vertices.emplace_back(vertex);
            updateCenterOfMass();
        }

        // 更新重心的函数
        void updateCenterOfMass() {
            glm::vec3 sum(0.0f);
            for (const auto& vertex : vertices) {
                sum += vertex;
            }
            centerOfMass = sum / static_cast<float>(vertices.size());
        }

        // 更新刚体状态的函数
        void update(float deltaTime) {
            // 使用牛顿第二定律更新线性速度和位置
            glm::vec3 acceleration = force / mass;
            velocity += acceleration * deltaTime;
            position += velocity * deltaTime;

            // 使用欧拉方程更新角速度和方向
            glm::vec3 angularAcceleration = inertia_tensor * torque;
            angular_velocity += angularAcceleration * deltaTime;
            glm::quat angularVelocityQuat(0.0f, angular_velocity.x, angular_velocity.y, angular_velocity.z);
            orientation += deltaTime * 0.5f * angularVelocityQuat * orientation;
            orientation = glm::normalize(orientation);

            // 清除力和扭矩以备下一次迭代
            force = glm::vec3(0.0f);
            torque = glm::vec3(0.0f);
        }

        // 应用力的函数
        void applyForce(const glm::vec3& f, const glm::vec3& point) {
            force += f;
            torque += glm::cross(point - position, f);
        }
    };

    //bool checkCollision(RigidBody& rb1, RigidBody& rb2) noexcept {
    //    // TODO
    //    return false;
    //}

    struct Ball : public RigidBody {
        const float radius = 50.0f;

        Ball() noexcept {
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
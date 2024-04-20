#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <memory>
#include <vector>

namespace jumping_ball::physics {
    class BoundingBox; // 前向声明
    class BoundingSphere; // 前向声明

    class BoundingVolume {
    public:
        virtual bool isIntersecting(const BoundingVolume& other) const = 0;
        virtual bool isIntersecting(const BoundingBox& box) const = 0;
        virtual bool isIntersecting(const BoundingSphere& sphere) const = 0;
        virtual void update(const glm::vec3& position, const glm::quat& orientation) = 0;
    };

    class BoundingBox : public BoundingVolume {
    public:
        glm::vec3 min; // 最小点
        glm::vec3 max; // 最大点

        BoundingBox(const glm::vec3& min, const glm::vec3& max)
            : min(min), max(max) {}

        bool isIntersecting(const BoundingVolume& other) const override {
            return other.isIntersecting(*this);
        }

        bool isIntersecting(const BoundingBox& box) const override {
            return (min.x <= box.max.x && max.x >= box.min.x) &&
                (min.y <= box.max.y && max.y >= box.min.y) &&
                (min.z <= box.max.z && max.z >= box.min.z);
        }

        bool isIntersecting(const BoundingSphere& sphere) const override;

        void update(const glm::vec3& position, const glm::quat& orientation) override {
            // 更新立方体的位置和方向
            // 这可能需要一些复杂的计算，取决于你的具体需求
        }
    };

    class BoundingSphere : public BoundingVolume {
    public:
        glm::vec3 center; // 球心
        float radius;     // 半径

        BoundingSphere(const glm::vec3& center, float radius)
            : center(center), radius(radius) {}

        bool isIntersecting(const BoundingVolume& other) const override {
            return other.isIntersecting(*this);
        }

        bool isIntersecting(const BoundingBox& box) const override;

        bool isIntersecting(const BoundingSphere& sphere) const override {
            float distance = glm::length(center - sphere.center);
            return distance < (radius + sphere.radius);
        }

        void update(const glm::vec3& position, const glm::quat& orientation) override {
            // 更新球体的位置
            center = position;
        }
    };

    class RigidBody {
    public:
        glm::vec3 position;                             // 位置
        glm::vec3 velocity;                             // 速度
        glm::vec3 angular_velocity;                     // 角速度
        float inverse_mass;                             // 质量的倒数
        glm::mat3 inertia_tensor;                       // 惯性张量
        glm::vec3 force;                                // 力
        glm::vec3 torque;                               // 扭矩
        glm::quat orientation;                          // 方向
        std::shared_ptr<std::vector<float>> vertices;   // 刚体的顶点
        glm::vec3 centerOfMass;                         // 重心

        // 碰撞体积
        std::vector<std::unique_ptr<BoundingVolume>> bounding_volumes;

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
}
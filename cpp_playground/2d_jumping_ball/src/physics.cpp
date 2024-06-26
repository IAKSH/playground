#include <jumping_ball/physics.hpp>

jumping_ball::physics::RigidBody::RigidBody() noexcept {
    vertices = std::make_shared<std::vector<float>>();
    applyDefaultValue();
}

jumping_ball::physics::RigidBody::RigidBody(std::shared_ptr<std::vector<float>> vertices) noexcept
    : vertices(vertices)
{
    applyDefaultValue();
}

void jumping_ball::physics::RigidBody::addVertex(float x,float y,float z) noexcept {
    vertices->resize(vertices->size() + 3);
    auto it = vertices->end() - 1;
    *it = z;
    *(--it) = y;
    *(--it) = z;
    updateCenterOfMass();
}

void jumping_ball::physics::RigidBody::updateCenterOfMass() noexcept {
    glm::vec3 sum(0.0f);
    for (const auto& vertex : *vertices) {
        sum += vertex;
    }
    centerOfMass = sum / static_cast<float>(vertices->size() * 3);
}

void jumping_ball::physics::RigidBody::update(float deltaTime) noexcept {
    // 使用牛顿第二定律更新线性速度和位置
    glm::vec3 acceleration = force * inverse_mass;
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

    // 更新碰撞体积
    for (auto& volume : bounding_volumes)
        volume->update(position, orientation);
}

void jumping_ball::physics::RigidBody::applyForce(const glm::vec3& f, const glm::vec3& point) noexcept {
    force += f;
    torque += glm::cross(point - position, f);
}

void jumping_ball::physics::RigidBody::applyDefaultValue() noexcept {
    position = glm::vec3(0.0f);
    velocity = glm::vec3(0.0f);
    angular_velocity = glm::vec3(0.0f);
    inverse_mass = 1.0f;
    inertia_tensor = glm::mat3(1.0f);
    force = glm::vec3(0.0f);
    torque = glm::vec3(0.0f);
    orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    centerOfMass = glm::vec3(0.0f);
}

// 在BoundingBox和BoundingSphere类的定义之后，实现它们的交叉检测方法
// 似乎是没有见过的处理类间循环依赖的方法

bool jumping_ball::physics::BoundingBox::isIntersecting(const BoundingSphere& sphere) const {
    glm::vec3 boxClosestPoint = glm::clamp(sphere.center, min, max);
    float distance = glm::length(boxClosestPoint - sphere.center);
    return distance < sphere.radius;
}

bool jumping_ball::physics::BoundingSphere::isIntersecting(const BoundingBox& box) const {
    return box.isIntersecting(*this);
}
#include <jumping_ball/gameobject.hpp>

jumping_ball::gameobject::GameObject::GameObject(std::shared_ptr<RenPipe> ren_pipe) noexcept
    : ren_pipe(ren_pipe)
{
    body = createDynamicSphere();
}

jumping_ball::gameobject::GameObject::~GameObject() noexcept {

}

glm::vec3 jumping_ball::gameobject::GameObject::getPosition() noexcept {
    btTransform trans;
    body->getMotionState()->getWorldTransform(trans);
    auto pos = trans.getOrigin();
    return glm::vec3(pos.x(), pos.y(), pos.z());
}

glm::vec3 jumping_ball::gameobject::GameObject::getVelocity() noexcept {
    auto vel = body->getLinearVelocity();
    return glm::vec3(vel.x(), vel.y(), vel.z());
}

glm::quat jumping_ball::gameobject::GameObject::getRotate() noexcept {
    btTransform trans;
    body->getMotionState()->getWorldTransform(trans);
    auto rotate = trans.getRotation();
    return glm::quat(rotate.x(), rotate.y(), rotate.z(), rotate.getW());
}

void jumping_ball::gameobject::GameObject::applyForce(const glm::vec3& target_position, float force_magnitude) noexcept {
    btVector3 body_position = body->getWorldTransform().getOrigin();
    btVector3 forceDirection = btVector3(target_position.x, target_position.y, target_position.z) - body_position;
    forceDirection.normalize();
    btVector3 force = forceDirection * btScalar(force_magnitude);
    body->applyCentralForce(force);
}

void jumping_ball::gameobject::GameObject::setCollisionCallback(std::function<void(GameObject&)> callback) noexcept {
    collision_callback = std::make_unique<CollisionCallback<GameObject>>(*this,callback);
}

void jumping_ball::gameobject::GameObject::checkCollision() noexcept {
    if (collision_callback) {
        physics::dynamics_world->contactTest(body, *collision_callback);
    }
}
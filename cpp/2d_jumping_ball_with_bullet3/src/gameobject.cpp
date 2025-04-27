#include <jumping_ball/gameobject.hpp>

namespace jumping_ball::gameobject {
    GameObject::GameObject(std::shared_ptr<RenPipe> ren_pipe,std::shared_ptr<RenObject> ren_obj) noexcept
        : ren_pipe(ren_pipe),ren_obj(ren_obj)
    {
        body = createDynamicSphere();
    }

    GameObject::~GameObject() noexcept {

    }

    glm::vec3 GameObject::getPosition() noexcept {
        btTransform trans;
        body->getMotionState()->getWorldTransform(trans);
        auto pos = trans.getOrigin();
        return glm::vec3(pos.x(), pos.y(), pos.z());
    }

    glm::vec3 GameObject::getVelocity() noexcept {
        auto vel = body->getLinearVelocity();
        return glm::vec3(vel.x(), vel.y(), vel.z());
    }

    glm::quat GameObject::getOrientation() noexcept {
        btTransform trans;
        body->getMotionState()->getWorldTransform(trans);
        auto rotate = trans.getRotation();
        return glm::quat(rotate.x(), rotate.y(), rotate.z(), rotate.getW());
    }

    void GameObject::applyForce(const glm::vec3& target_position, float force_magnitude) noexcept {
        btVector3 body_position = body->getWorldTransform().getOrigin();
        btVector3 forceDirection = btVector3(target_position.x, target_position.y, target_position.z) - body_position;
        forceDirection.normalize();
        btVector3 force = forceDirection * btScalar(force_magnitude);
        body->applyCentralForce(force);
    }

    void GameObject::setCollisionCallback(std::function<void(GameObject&)> callback) noexcept {
        collision_callback = std::make_unique<CollisionCallback<GameObject>>(*this,callback);
    }

    void GameObject::checkCollision() noexcept {
        if (collision_callback) {
            physics::dynamics_world->contactTest(body, *collision_callback);
        }
    }

    void GameObject::draw() noexcept {
        // 应用自身position和rotation
        // 因为ren_obj可能共用，所以只能在绘制时才更新position和rotation
        ren_obj->position = getPosition();
        ren_obj->orientation = getOrientation();
        ren_pipe->draw(*ren_obj);
    }
}
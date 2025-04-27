#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <bullet/btBulletDynamicsCommon.h>
#include <memory>
#include <vector>

namespace jumping_ball::physics {
    extern btDefaultCollisionConfiguration* collision_configuration;
    extern btCollisionDispatcher* dispatcher;
    extern btBroadphaseInterface* overlapping_pair_cache;
    extern btSequentialImpulseConstraintSolver* solver;
    extern btDiscreteDynamicsWorld* dynamics_world;
    extern btAlignedObjectArray<btCollisionShape*> collision_shapes;

    void initialize() noexcept;
    void uninitialize() noexcept;
    btRigidBody* createHullFront() noexcept;
    btRigidBody* createHullBack() noexcept;
    btRigidBody* createHullLeft() noexcept;
    btRigidBody* createHullRight() noexcept;
    btRigidBody* createHullUp() noexcept;
    btRigidBody* createHullDown() noexcept;
    btRigidBody* createDynamicSphere() noexcept;
    void processStepSimulation(float delta_time) noexcept;
}
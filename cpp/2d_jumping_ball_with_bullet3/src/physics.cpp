#include <jumping_ball/physics.hpp>

btDefaultCollisionConfiguration* jumping_ball::physics::collision_configuration;
btCollisionDispatcher* jumping_ball::physics::dispatcher;
btBroadphaseInterface* jumping_ball::physics::overlapping_pair_cache;
btSequentialImpulseConstraintSolver* jumping_ball::physics::solver;
btDiscreteDynamicsWorld* jumping_ball::physics::dynamics_world;
btAlignedObjectArray<btCollisionShape*> jumping_ball::physics::collision_shapes;

void jumping_ball::physics::initialize() noexcept {
    collision_configuration = new btDefaultCollisionConfiguration();
    dispatcher = new btCollisionDispatcher(collision_configuration);
    overlapping_pair_cache = new btDbvtBroadphase();
    solver = new btSequentialImpulseConstraintSolver();
    dynamics_world = new btDiscreteDynamicsWorld(dispatcher, overlapping_pair_cache, solver, collision_configuration);

    // 临时设置重力方向为y轴负方向
    dynamics_world->setGravity(btVector3(0, -10, 0));
    //dynamics_world->setGravity(btVector3(0, -0.01, 0));
}

void jumping_ball::physics::uninitialize() noexcept {
    // cleanup in the reverse order of creation/initialization
    // remove the rigid bodies from the dynamics world and delete them
    for (int i = dynamics_world->getNumCollisionObjects() - 1; i >= 0; i--) {
        btCollisionObject* obj = dynamics_world->getCollisionObjectArray()[i];
        btRigidBody* body = btRigidBody::upcast(obj);
        if (body && body->getMotionState()) {
            delete body->getMotionState();
        }
        dynamics_world->removeCollisionObject(obj);
        delete obj;
    }

    // delete collision shapes
    for (int i = 0; i < collision_shapes.size(); i++) {
        btCollisionShape* shape = collision_shapes[i];
        collision_shapes[i] = nullptr;
        delete shape;
    }

    // delete other global values
    delete dynamics_world;
    delete solver;
    delete overlapping_pair_cache;
    delete dispatcher;
    delete collision_configuration;
    
    // actually btAlignedObjectArray can RAII
    // if collision_shapes is a local var, we can just leave it
    collision_shapes.clear();
}

btRigidBody* jumping_ball::physics::createHullFront() noexcept {
    btCollisionShape* collision_shape = new btBoxShape(btVector3(400, 400, 10));
    collision_shapes.push_back(collision_shape);

    btDefaultMotionState* motion_state = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 0, 60)));

    btRigidBody::btRigidBodyConstructionInfo rb_info(0, motion_state, collision_shape, btVector3(0, 0, 0));

    btRigidBody* body = new btRigidBody(rb_info);
    body->setCollisionFlags(body->getCollisionFlags() | btCollisionObject::CF_STATIC_OBJECT);

    body->setRestitution(0.9f); // 设置恢复系数为0.9
    body->setFriction(0.25f);
    body->setRollingFriction(0.5f);

    dynamics_world->addRigidBody(body);

    return body;
}

btRigidBody* jumping_ball::physics::createHullBack() noexcept {
    btCollisionShape* collision_shape = new btBoxShape(btVector3(400, 400, 10));
    collision_shapes.push_back(collision_shape);

    btDefaultMotionState* motion_state = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 0, -60)));

    btRigidBody::btRigidBodyConstructionInfo rb_info(0, motion_state, collision_shape, btVector3(0, 0, 0));

    btRigidBody* body = new btRigidBody(rb_info);
    body->setCollisionFlags(body->getCollisionFlags() | btCollisionObject::CF_STATIC_OBJECT);

    body->setRestitution(0.9f); // 设置恢复系数为0.9
    body->setFriction(0.25f);
    body->setRollingFriction(0.5f);

    dynamics_world->addRigidBody(body);

    return body;
}

btRigidBody* jumping_ball::physics::createHullLeft() noexcept {
    btCollisionShape* collision_shape = new btBoxShape(btVector3(10, 400, 400));
    collision_shapes.push_back(collision_shape);

    btDefaultMotionState* motion_state = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(-410, 0, 0)));

    btRigidBody::btRigidBodyConstructionInfo rb_info(0, motion_state, collision_shape, btVector3(0, 0, 0));

    btRigidBody* body = new btRigidBody(rb_info);
    body->setCollisionFlags(body->getCollisionFlags() | btCollisionObject::CF_STATIC_OBJECT);

    body->setRestitution(0.9f); // 设置恢复系数为0.9
    body->setFriction(0.25f);
    body->setRollingFriction(0.5f);

    dynamics_world->addRigidBody(body);

    return body;
}

btRigidBody* jumping_ball::physics::createHullRight() noexcept {
    btCollisionShape* collision_shape = new btBoxShape(btVector3(10, 400, 400));
    collision_shapes.push_back(collision_shape);

    btDefaultMotionState* motion_state = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(410, 0, 0)));

    btRigidBody::btRigidBodyConstructionInfo rb_info(0, motion_state, collision_shape, btVector3(0, 0, 0));

    btRigidBody* body = new btRigidBody(rb_info);
    body->setCollisionFlags(body->getCollisionFlags() | btCollisionObject::CF_STATIC_OBJECT);

    body->setRestitution(0.9f); // 设置恢复系数为0.9
    body->setFriction(0.25f);
    body->setRollingFriction(0.5f);

    dynamics_world->addRigidBody(body);

    return body;
}

btRigidBody* jumping_ball::physics::createHullUp() noexcept {
    btCollisionShape* collision_shape = new btBoxShape(btVector3(400, 10, 400));
    collision_shapes.push_back(collision_shape);

    btDefaultMotionState* motion_state = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 410, 0)));

    btRigidBody::btRigidBodyConstructionInfo rb_info(0, motion_state, collision_shape, btVector3(0, 0, 0));

    btRigidBody* body = new btRigidBody(rb_info);
    body->setCollisionFlags(body->getCollisionFlags() | btCollisionObject::CF_STATIC_OBJECT);

    body->setRestitution(0.9f); // 设置恢复系数为0.9
    body->setFriction(0.25f);
    body->setRollingFriction(0.5f);

    dynamics_world->addRigidBody(body);

    return body;
}

btRigidBody* jumping_ball::physics::createHullDown() noexcept {
    btCollisionShape* collision_shape = new btBoxShape(btVector3(400, 10, 400));
    collision_shapes.push_back(collision_shape);

    btDefaultMotionState* motion_state = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, -410, 0)));

    btRigidBody::btRigidBodyConstructionInfo rb_info(0, motion_state, collision_shape, btVector3(0, 0, 0));

    btRigidBody* body = new btRigidBody(rb_info);
    body->setCollisionFlags(body->getCollisionFlags() | btCollisionObject::CF_STATIC_OBJECT);

    body->setRestitution(0.9f); // 设置恢复系数为0.9
    body->setFriction(0.25f);
    body->setRollingFriction(0.5f);

    dynamics_world->addRigidBody(body);

    return body;
}


// 3D
/*
btRigidBody* jumping_ball::physics::createDynamicSphere() noexcept {
    btCollisionShape* col_shape = new btSphereShape(25.0f);
    collision_shapes.push_back(col_shape);

    btTransform start_transform;
    start_transform.setIdentity();

    constexpr btScalar mass(1.0f);
    constexpr bool is_dynamic = (mass != 0.0f);

    btVector3 local_inertia(0.0f, 0.0f, 0.0f);
    if (is_dynamic) {
        col_shape->calculateLocalInertia(mass, local_inertia);
    }

    start_transform.setOrigin(btVector3(2.0f, 10.0f, 0.0f));

    btDefaultMotionState* motion_state = new btDefaultMotionState(start_transform);
    btRigidBody::btRigidBodyConstructionInfo rb_info(mass, motion_state, col_shape, local_inertia);
    btRigidBody* body = new btRigidBody(rb_info);

    dynamics_world->addRigidBody(body);

    return body;
}
*/

// 2D
btRigidBody* jumping_ball::physics::createDynamicSphere() noexcept {
    btCollisionShape* col_shape = new btSphereShape(25.0f);
    collision_shapes.push_back(col_shape);

    btTransform start_transform;
    start_transform.setIdentity();

    constexpr btScalar mass(1.0f);
    constexpr bool is_dynamic = (mass != 0.0f);

    btVector3 local_inertia(0.0f, 0.0f, 0.0f);
    if (is_dynamic) {
        col_shape->calculateLocalInertia(mass, local_inertia);
    }

    start_transform.setOrigin(btVector3(2.0f, 10.0f, 0.0f));

    btDefaultMotionState* motion_state = new btDefaultMotionState(start_transform);
    btRigidBody::btRigidBodyConstructionInfo rb_info(mass, motion_state, col_shape, local_inertia);
    btRigidBody* body = new btRigidBody(rb_info);

    // 限制Z轴上的线性和角动量
    body->setLinearFactor(btVector3(1, 1, 0));
    body->setAngularFactor(btVector3(1, 1, 0));

    body->setRestitution(0.95f);
    body->setFriction(0.25f);
    body->setRollingFriction(0.5f);

    dynamics_world->addRigidBody(body);

    return body;
}


void jumping_ball::physics::processStepSimulation(float delta_time) noexcept {
    //float time_step = delta_time / 1000.0f;  // convert ms to s
    float time_step = delta_time / 50.0f;
    int max_substeps = 10;
    float fixed_time_step = 1.0f / 60.0f;  // 60 Hz
    dynamics_world->stepSimulation(time_step, max_substeps, fixed_time_step);
}

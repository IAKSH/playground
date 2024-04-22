#include <jumping_ball/misc.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/quaternion_trigonometric.hpp>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

namespace jumping_ball::misc {
    Point::Point() noexcept
        : xyz{0.0f,0.0f,0.0f} {}

    Point::Point(const float& x,const float& y,const float& z) noexcept
        : xyz{x,y,z} {}

    RotatablePoint::RotatablePoint() noexcept
        : right{1.0f,0.0f,0.0f},up{0.0f,1.0f,0.0f}
    {
        glm::quat quat(glm::vec3(0.0f,0.0f,-1.0f));
        orientation = {quat[0],quat[1],quat[2],quat[3]};
        updateVectors();
    }

    float RotatablePoint::getYaw() const noexcept
    {
        glm::quat quat(orientation[0],orientation[1],orientation[2],orientation[3]);
        return glm::eulerAngles(quat)[0];
    }

    float RotatablePoint::getPitch() const noexcept
    {
        glm::quat quat(orientation[0],orientation[1],orientation[2],orientation[3]);
        return glm::eulerAngles(quat)[1];
    }

    float RotatablePoint::getRoll() const noexcept
    {
        glm::quat quat(orientation[0],orientation[1],orientation[2],orientation[3]);
        return glm::eulerAngles(quat)[2];
    }

    void RotatablePoint::rotate(float dUp,float dRight,float dRoll) noexcept
    {
        glm::quat quat(orientation[0],orientation[1],orientation[2],orientation[3]);
        glm::vec3 vecRight(right[0],right[1],right[2]);
        glm::vec3 vecUp(up[0],up[1],up[2]);

        glm::quat yawQuat = glm::angleAxis(glm::radians(dUp), vecRight);
        glm::quat pitchQuat = glm::angleAxis(glm::radians(-dRight), vecUp);
        quat = yawQuat * pitchQuat * quat;

        orientation = {quat[0],quat[1],quat[2],quat[3]};
        right = {vecRight[0],vecRight[1],vecRight[2]};
        up = {vecUp[0],vecUp[1],vecUp[2]};

        updateVectors();
    }

    void RotatablePoint::updateVectors() noexcept
    {
        glm::quat quat(orientation[0],orientation[1],orientation[2],orientation[3]);
        glm::vec3 vecRight(right[0],right[1],right[2]);
        glm::vec3 vecUp(up[0],up[1],up[2]);

        glm::vec3 direction = glm::normalize(glm::rotate(quat, glm::vec3(0.0f, 0.0f, -1.0f)));
        vecRight = glm::normalize(glm::cross(direction, vecUp));
        vecUp = glm::normalize(glm::cross(vecRight, direction));

        orientation = {quat[0],quat[1],quat[2],quat[3]};
        right = {vecRight[0],vecRight[1],vecRight[2]};
        up = {vecUp[0],vecUp[1],vecUp[2]};
    }

    void RotatablePoint::move(float dFront,float dRight,float dHeight) noexcept
    {
        glm::quat quat(orientation[0],orientation[1],orientation[2],orientation[3]);
        glm::vec3 vecRight(right[0],right[1],right[2]);
        glm::vec3 vecUp(up[0],up[1],up[2]);
        glm::vec3 position(x,y,z);

        position += dFront * glm::rotate(quat, glm::vec3(0.0f, 0.0f, -1.0f)) + dRight * vecRight + dHeight * vecUp;

        x = position[0];
        y = position[1];
        z = position[2];
    }
}
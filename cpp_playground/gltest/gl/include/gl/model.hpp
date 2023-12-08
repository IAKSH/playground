// Mesh,Model
// (VAO,VBO,EBO)

#include <glad/gles2.h>
#include <glm/matrix.hpp>
#include <vector>

#include <gl/model.hpp>

namespace nioes {
    class Position {
    private:
        glm::vec3 xyz;
        
    public:
        Position(float x,float y,float z) noexcept;
        Position(glm::vec3 xyz) noexcept;
        Position(Position&) = default;
        Position() = default;
        ~Position() = default;

        float& x{xyz.x};
        float& y{xyz.y};
        float& z{xyz.z};

        float distance(const Position& pos) const noexcept;
        Position operator+ (const Position& pos) const noexcept;
        Position operator- (const Position& pos) const noexcept;
        bool operator== (const Position& pos) const noexcept;
        void operator= (const Position& pos) noexcept;
        void operator+= (const Position& pos) noexcept;
        void operator-=(const Position* pos) noexcept;
    };

    class Rotator {
    public:
        Rotator(float y,float p,float r) noexcept;
        Rotator(glm::vec3 ypr) noexcept;
        Rotator(Rotator&) = default;
        Rotator() = default;
        ~Rotator() = default;

        // 欧拉角
        glm::vec3 ypr;
        float& y{ypr.x};
        float& p{ypr.y};
        float& r{ypt.z};

        Position operator+ (const Rotator& pos) const noexcept;
        Position operator- (const Rotator& pos) const noexcept;
        bool operator== (const Rotator& pos) const noexcept;
        void operator= (const Rotator& pos) noexcept;
        void operator+= (const Rotator& pos) noexcept;
        void operator-=(const Rotator* pos) noexcept;
        
        glm::vec3 get_up_vec() noexcept;
        glm::vec3 get_right_vec() noexcept;
        glm::vec3 get_front_vec() noexcept;
    };

    // 模型资源，应当全局唯一
    class Model {
    private:
        struct Mesh {
            // TODO:
        };
        std::vector<Mesh> meshes;
    };

    // 骨骼动画资源，应当全局唯一
    // 动画进行状态储存在Object，而不是这里
    class Animation {

    };

    // TODO: 可能需要为模型与动画提供工厂类，或者是工厂函数
    // 以提供Image（Texture）以及其他类似的，可以复用的资源的缓存

    // 模型，材质和动画的载体
    class Object {
    private:
        // TODO: 模型，材质和动画

    public:
        Object() = default;
        Object(Object&) = default;
        ~Object() = default;

        Position pos;
        Rotator rotate;
    };

    class Camera : public Object {
    
    };
}
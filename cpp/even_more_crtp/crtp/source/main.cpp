// CRTP用于在编译期静态实现的一种功能（或语义）注入
//    通过将通用逻辑编写在CRTP基类中，子类可以只关注自身的基础数据
//    而基于这些数据的操作都通过继承CRTP基类，在CRTP基类中统一实现
//    从而具有很高的标准性（接口一致性）和复用率
// 而多态（虚表）用于实现对某一系列类型的动态的统一管理
//    多态主要在乎的是这些类会不会被在逻辑上视为一种东西进行统一管理，
//    而对于一些逻辑上并没有统一管理必要，但是依然需要高度标准化和代码复用的场景
//    就应该使用CRTP
// 二者不是替代关系，该用什么用什么

/*
还应该提到的是，传统的没有任何virtual的继承，其实也是静态分派，性能与CRTP方案无异，甚至更不容易出现二进制膨胀
    但是CRTP可以做（或者说更适合做）行为注入，
    并且可以依赖模板元编程使用if constexpr依据Derived的不同，静态地采用不同的策略
    而传统继承只能在运行时动态地做到这一点

    CRTP的目的是：基类定义行为（同时也可以有语义），而子类提供数据或语义，在继承的过程中，基类将行为和语义插入了子类中
    传统继承是：基类定义接口，派生类继承并扩展行为。

    简单地说，虽然有很多相似处，但CRTP更像是我们理想中的，在编译期静态完成的，几乎0（运行时）开销的高度代码复用（或者说模块化？）
*/

/*
还有一种更极端的路子，直接完全放弃动态分派，在编译期用模板展开所有可能（案例：fast_io）
    等到哪天反射提案落地了，这种写法可能会变得容易阅读许多，但是始终会有二进制膨胀的问题
    也许适用于一些动态性不那么高的地方，可以提供更高的性能
*/

/*
以Game开发为例，下面的这套骨架可以比较方便地扩充出ECS（当然，通过CRTP和虚表）
    和脚本系统（比如修改update和receive方法，调用对应的脚本解释器并传入相应的回调函数和他们在脚本中的映射）。
*/

#include <iostream>
#include <vector>
#include <memory>
#include <concepts>
#include <cmath>
#include <queue>
#include <string>

// ------------------- Concepts -------------------

template<typename T>
concept WithReadableWritableX = requires(T t, float f) {
    { t.get_x() } -> std::convertible_to<float>;
    { t.set_x(f) } -> std::same_as<void>;
};

template<typename T>
concept WithReadableWritableY = requires(T t, float f) {
    { t.get_y() } -> std::convertible_to<float>;
    { t.set_y(f) } -> std::same_as<void>;
};

template<typename T>
concept WithReadable2DPos = WithReadableWritableX<T> && WithReadableWritableY<T>;

template<typename T>
concept WithHealthPoint = requires(T t,float f) {
    { t.health() } -> std::convertible_to<float>;
};

// ------------------- CRTP Positionable -------------------

template<typename Derived>
class Positionable {
public:
    // 如果CRTP基类中依然使用requires对整个类进行检查，会导致requires检查早于子类的类型定义，从而导致检查失败
    // 在CRTP基类中使用static_assert也不行，原因同上
    // 只能在CRTP基类的每个函数声明中进行requires，比如下面这样
    void move_by(float dx, float dy)
        requires WithReadable2DPos<Derived> {
        derived().set_x(derived().get_x() + dx);
        derived().set_y(derived().get_y() + dy);
    }

    float distance_to(float x, float y) const
        requires WithReadable2DPos<Derived> {
        float dx = derived().get_x() - x;
        float dy = derived().get_y() - y;
        return std::sqrt(dx * dx + dy * dy);
    }

protected:
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }
};

// ------------------- Message System -------------------

struct Message {
    virtual ~Message() = default;
    virtual std::size_t type_id() const = 0;
};

// 模板类的一种奇妙用法之会自动分配的static_type_id
template<typename T>
struct MessageBase : Message {
    static std::size_t static_type_id() {
        //static char dummy;
        // 注意获取的是地址，不同的模板类实例的static变量当然有不同的地址
        //return reinterpret_cast<std::size_t>(&dummy);
        
        // 离谱的是，甚至可以用这个函数的地址
        return reinterpret_cast<std::size_t>(&static_type_id);
    }
    std::size_t type_id() const override {
        return static_type_id();
    }
};

// 攻击消息
struct AttackMessage : public MessageBase<AttackMessage> {
    float target_x;
    float target_y;
    float radius;
    int damage;

    AttackMessage(float x, float y, float r, int d)
        : target_x(x), target_y(y), radius(r), damage(d) {}
};

// 治疗消息
struct HealMessage : public MessageBase<HealMessage> {
    float target_x;
    float target_y;
    float radius;
    int heal;

    HealMessage(float x, float y, float r, int h)
        : target_x(x), target_y(y), radius(r), heal(h) {}
};

class MessageQueue {
    std::queue<std::unique_ptr<Message>> messages;

public:
    void push(std::unique_ptr<Message> msg) {
        messages.push(std::move(msg));
    }

    bool empty() const { return messages.empty(); }

    std::unique_ptr<Message> pop() {
        if (messages.empty()) return nullptr;
        auto msg = std::move(messages.front());
        messages.pop();
        return msg;
    }
};

// ------------------- IEntity Interface -------------------

class IEntity {
public:
    virtual void update(MessageQueue& queue) = 0;
    virtual void receive(const Message& msg) = 0;
    virtual ~IEntity() = default;
};

// ------------------- Player -------------------

// 以Player为例，这里将Player和PlayerEntity分开了，这种设计可以让：
//     Player 只关心数据和行为（大部分由CRTP基类注入，自身只维护自己的数据及其读写），是一个纯粹的 domain object
//     PlayerEntity 关心系统接口，即实现 IEntity，参与消息分发和调度
//
//     Player 甚至可以脱离 IEntity 系统，参与一些独立的数值计算，或者某种其他东西

class Player : public Positionable<Player> {
    float x = 0.0f;
    float y = 0.0f;

public:
    float get_x() const { return x; }
    float get_y() const { return y; }
    void set_x(float val) { x = val; }
    void set_y(float val) { y = val; }
};

class PlayerEntity : public Player, public IEntity {
protected:
    void attack(MessageQueue& queue, float tx, float ty) {
        queue.push(std::make_unique<AttackMessage>(tx, ty, 5.0f, 10));
    }

    void heal(MessageQueue& queue, float tx, float ty) {
        queue.push(std::make_unique<HealMessage>(tx, ty, 5.0f, 5));
    }

public:
    void update(MessageQueue& queue) override {
        move_by(1.0f, 0.0f); // Move right
        attack(queue, get_x() + 2.0f, get_y()); // Attack ahead
        heal(queue, get_x(), get_y()); // Heal around self
    }

    void receive(const Message&) override {
        // Player doesn't react to messages in this demo
    }
};

// ------------------- Enemy -------------------

class Enemy : public Positionable<Enemy> {
    float x = 0.0f;
    float y = 0.0f;
    int hp = 100;

public:
    float get_x() const { return x; }
    float get_y() const { return y; }
    void set_x(float val) { x = val; }
    void set_y(float val) { y = val; }

    void take_damage(int dmg) {
        hp -= dmg;
        std::cout << "Enemy at (" << x << ", " << y << ") took " 
                  << dmg << " damage. HP = " << hp << "\n";
    }

    void heal_up(int h) {
        hp += h;
        std::cout << "Enemy at (" << x << ", " << y << ") healed " 
                  << h << ". HP = " << hp << "\n";
    }
};

class EnemyEntity : public Enemy, public IEntity {
public:
    EnemyEntity(float x = 0.0f,float y = 0.0f) {
        set_x(x);
        set_y(y);
    }

    void update(MessageQueue&) override {
        // Enemy doesn't act in this demo
    }

    void receive(const Message& msg) override {
        if (msg.type_id() == AttackMessage::static_type_id()) {
            auto& attack = static_cast<const AttackMessage&>(msg);
            if (distance_to(attack.target_x, attack.target_y) <= attack.radius) {
                take_damage(attack.damage);
            }
        } else if (msg.type_id() == HealMessage::static_type_id()) {
            auto& heal = static_cast<const HealMessage&>(msg);
            if (distance_to(heal.target_x, heal.target_y) <= heal.radius) {
                heal_up(heal.heal);
            }
        }
    }
};

// ------------------- Main Simulation -------------------

int main() {
    MessageQueue queue;

    std::vector<std::unique_ptr<IEntity>> entities;
    entities.emplace_back(std::make_unique<PlayerEntity>());
    entities.emplace_back(std::make_unique<EnemyEntity>(3.0f,0.5f));
    entities.emplace_back(std::make_unique<EnemyEntity>(8.0f,2.0f));

    std::cout << "=== Simulation Start ===\n";

    // Step 1: Update all entities (player moves and attacks)
    for (auto& e : entities) {
        e->update(queue);
    }

    // Step 2: Dispatch messages
    while (!queue.empty()) {
        auto msg = queue.pop();
        for (auto& e : entities) {
            e->receive(*msg);
        }
    }

    std::cout << "=== Simulation End ===\n";
    return 0;
}

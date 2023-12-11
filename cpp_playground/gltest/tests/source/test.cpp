#include <nioes/instance.hpp>

int main() noexcept {
    nioes::Instance instance("instance test",800,600);
    while(true) {
        instance.flush();
        nioes::Instance::update_all();
    }
}
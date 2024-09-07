#include <inject_dll.h>
#include <Windows.h>
#include <iostream>
#include <thread>

const char* get_dll_info() noexcept {
	return "a dll made for injection";
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        std::cout << "DLL_PROCESS_ATTACH\n";
        std::thread([]() {
            unsigned int i = 0;
            while (true) {
                std::cout << "dll injected!!!\ti = " << ++i << '\n';
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            }).detach();
        break;
    case DLL_PROCESS_DETACH:
        std::cout << "DLL_PROCESS_DETACH\n";
        break;
    }
    return TRUE;
}
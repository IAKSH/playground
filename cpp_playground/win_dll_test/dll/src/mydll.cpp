#include <mydll.h>
#include <iostream>
#include <Windows.h>

void say(const char* str) {
	std::cout << str;
}

void say_hello() {
	std::cout << "hello\n";
}

const char* get_dll_info() {
	return "a common dll for test";
}

float add(float m, float n) {
	return m + n;
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        std::cout << "DLL_PROCESS_ATTACH\n";
        break;
    case DLL_PROCESS_DETACH:
        std::cout << "DLL_PROCESS_DETACH\n";
        break;
    }
    return TRUE;
}
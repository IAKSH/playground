// 这里的if宏是为了兼顾dll自身的编译以及dll使用者的编译
// dll编译时通过构建系统定义MYDLL_EXPORTS，以启用__declspec(dllexport)，将下面的api导出
// 使用者编译时未定义MYDLL_EXPORTS，则使用__declspec(dllimport)，将api导入
// 虽然文档是这么写的，但是似乎对于msvc17，即使dll编译时没有定义这个宏似乎也是可以的，不知道为什么

#ifdef MYDLL_EXPORTS
#define MYDLL_API __declspec(dllexport)
#else
#define MYDLL_API __declspec(dllimport)
#endif

extern "C" MYDLL_API void say(const char* str);
extern "C" MYDLL_API void say_hello();
extern "C" MYDLL_API const char* get_dll_info();
extern "C" MYDLL_API float add(float m, float n);

// 看起来是MSVC背后完成了导出函数和导出表（利用上面的宏里的东西）
// 如果不带这个，会导致调用的时候找不到符号
// 这么一来，就和linux那边比较统一了，除了多了个dll main?
// 不过似乎linux的.so可以用__attribute__((constructor))和__attribute__((destructor))
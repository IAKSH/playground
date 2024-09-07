// �����if����Ϊ�˼��dll����ı����Լ�dllʹ���ߵı���
// dll����ʱͨ������ϵͳ����MYDLL_EXPORTS��������__declspec(dllexport)���������api����
// ʹ���߱���ʱδ����MYDLL_EXPORTS����ʹ��__declspec(dllimport)����api����
// ��Ȼ�ĵ�����ôд�ģ������ƺ�����msvc17����ʹdll����ʱû�ж���������ƺ�Ҳ�ǿ��Եģ���֪��Ϊʲô

#ifdef MYDLL_EXPORTS
#define MYDLL_API __declspec(dllexport)
#else
#define MYDLL_API __declspec(dllimport)
#endif

extern "C" MYDLL_API void say(const char* str);
extern "C" MYDLL_API void say_hello();
extern "C" MYDLL_API const char* get_dll_info();
extern "C" MYDLL_API float add(float m, float n);

// ��������MSVC��������˵��������͵�������������ĺ���Ķ�����
// �������������ᵼ�µ��õ�ʱ���Ҳ�������
// ��ôһ�����ͺ�linux�Ǳ߱Ƚ�ͳһ�ˣ����˶��˸�dll main?
// �����ƺ�linux��.so������__attribute__((constructor))��__attribute__((destructor))
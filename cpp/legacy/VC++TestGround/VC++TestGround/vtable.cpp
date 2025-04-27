/*
#include <chrono>
#include <iostream>
#include <memory>

static int i = 0;

struct VTableBase
{
    virtual void doit() = 0;
    virtual void foo() = 0;
};

struct VTableDiverse0 : public VTableBase
{
    virtual void doit() override
    {
        i++;
    }

    virtual void foo() override
    {
        std::cout << "vtable diverse 0 : foo\n";
    }
};

struct VTableDiverse1 : public VTableBase
{
    virtual void doit() override
    {
        std::cout << "vtable diverse 1 : doit\n";
    }

    virtual void foo() override
    {
        std::cout << "vtable diverse 1 : foo\n";
    }
};

struct VtableDiverse1Diverse : public VTableDiverse1
{
    virtual void doit() override
    {
        std::cout << "vtable diverse 1 diverse : doit\n";
    }

    virtual void foo() override
    {
        std::cout << "vtable diverse 1 diverse : foo\n";
    }
};

struct VtableDiverse0Diverse : public VTableDiverse0
{
    virtual void doit() override
    {
        std::cout << "vtable diverse 0 diverse : doit\n";
    }

    virtual void foo() override
    {
        std::cout << "vtable diverse 0 diverse : foo\n";
    }
};

template <typename Diverse>
struct CRTPBase
{
    CRTPBase() = default;
    ~CRTPBase() = default;
    void doit() { static_cast<Diverse*>(this)->imp_doit(); }
    void foo() { static_cast<Diverse*>(this)->imp_foo(); }
};

struct CRTPDiverse0 : CRTPBase<CRTPDiverse0>
{
    CRTPDiverse0() = default;
    ~CRTPDiverse0() = default;
    void imp_doit()
    {
        std::cout << "CRTP diverse 0 : imp_doit\n";
    }

    void imp_foo()
    {
        std::cout << "CRTP diverse 0 : imp_foo\n";
    }
};

struct CRTPDiverse1 : CRTPBase<CRTPDiverse1>
{
    CRTPDiverse1() = default;
    ~CRTPDiverse1() = default;
    void imp_doit()
    {
        i++;
    }

    void imp_foo()
    {
        std::cout << "CRTP diverse 1 : imp_foo\n";
    }
};

int main()
{
    VTableDiverse0 vt0;
    VTableBase* vtb = &vt0;
    auto vtableBegin = std::chrono::high_resolution_clock::now();
    vtb->doit();
}
*/
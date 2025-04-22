#include <trick/trick_base.hpp>
#define UNICODE
#include <windows.h>
#include <tchar.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <iostream>
#include <thread>

namespace trick::win {
class Bitmap : public base::Bitmap<Bitmap> {
public:
    explicit Bitmap(HBITMAP hbitmap)
        : hbitmap(hbitmap) {}

    Bitmap(Bitmap&) = default;

    int width_impl() const {
        BITMAP bmp;
        if (GetObject(hbitmap, sizeof(BITMAP), &bmp) == 0) {
            throw std::runtime_error("Failed to get bitmap object.");
        }
        return bmp.bmWidth;
    }

    int height_impl() const {
        BITMAP bmp;
        if (GetObject(hbitmap, sizeof(BITMAP), &bmp) == 0) {
            throw std::runtime_error("Failed to get bitmap object.");
        }
        return bmp.bmHeight;
    }

    char* data_impl() const {
        BITMAP bmp;
        if (GetObject(hbitmap, sizeof(BITMAP), &bmp) == 0) {
            throw std::runtime_error("Failed to get bitmap object.");
        }

        int dataSize = bmp.bmWidthBytes * bmp.bmHeight;
        char* buffer = new char[dataSize];
        if (!GetBitmapBits(hbitmap, dataSize, buffer)) {
            delete[] buffer;
            throw std::runtime_error("Failed to get bitmap bits.");
        }
        return buffer;
    }

    void save_to_file_impl(std::string_view path) const {
        BITMAPFILEHEADER fileHeader = {};
        BITMAPINFOHEADER infoHeader = {};
        BITMAP bmp;

        if (GetObject(hbitmap, sizeof(BITMAP), &bmp) == 0) {
            throw std::runtime_error("Failed to get bitmap object.");
        }

        int dataSize = bmp.bmWidthBytes * bmp.bmHeight;
        std::vector<char> buffer(dataSize);
        if (!GetBitmapBits(hbitmap, dataSize, buffer.data())) {
            throw std::runtime_error("Failed to get bitmap bits.");
        }

        fileHeader.bfType = 0x4D42; // "BM"
        fileHeader.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + dataSize;
        fileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

        infoHeader.biSize = sizeof(BITMAPINFOHEADER);
        infoHeader.biWidth = bmp.bmWidth;
        infoHeader.biHeight = -bmp.bmHeight;
        infoHeader.biPlanes = 1;
        infoHeader.biBitCount = bmp.bmBitsPixel;
        infoHeader.biCompression = BI_RGB;
        infoHeader.biSizeImage = dataSize;

        std::ofstream file(path.data(), std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file for writing.");
        }

        file.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
        file.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));
        file.write(buffer.data(), dataSize);
    }

    const HBITMAP& bitmap() const {
        return hbitmap;
    }

private:
    HBITMAP hbitmap;
};

class ScreenRecorder : public base::ScreenRecorder<ScreenRecorder,Bitmap> {
public:
    ScreenRecorder() {
        auto reso = getScreenResolution(getPrimaryMonitor());
        width = reso.first;
        height = reso.second;
        init();
    }

    ScreenRecorder(int w,int h) : width(w), height(h) {
        init();
    }

    ScreenRecorder(ScreenRecorder&) = default;

    ScreenRecorder(const ScreenRecorder&) = delete;
    ScreenRecorder& operator=(const ScreenRecorder&) = delete;

    ~ScreenRecorder() {
        DeleteObject(bitmap);
        DeleteDC(memoryDC);
        ReleaseDC(nullptr,screenDC);
    }

    std::shared_ptr<base::Bitmap<Bitmap>> capture_impl() {
        HDC screen_dc = GetDC(nullptr);
        BitBlt(memoryDC, 0, 0, width, height, screen_dc, 0, 0, SRCCOPY);
        ReleaseDC(nullptr, screen_dc);

        return std::make_shared<Bitmap>(bitmap);
    }

private:
    int width,height;
    HBITMAP bitmap;
    HDC screenDC;
    HDC memoryDC;

    void init() {
        screenDC = GetDC(nullptr);
        memoryDC = CreateCompatibleDC(screenDC);
        bitmap = CreateCompatibleBitmap(screenDC,width,height);
        SelectObject(memoryDC,bitmap);
    }

    static HMONITOR getPrimaryMonitor() {
        POINT ptZero = {0,0};
        return MonitorFromPoint(ptZero,MONITOR_DEFAULTTOPRIMARY);
    }

    static std::pair<float,float> getScreenResolution(HMONITOR monitor) {
        MONITORINFOEX info = {};
        info.cbSize = sizeof(info);
        GetMonitorInfo(monitor,&info);
        DEVMODE devmode = {};
        devmode.dmSize = sizeof(DEVMODE);
        EnumDisplaySettings(info.szDevice,ENUM_CURRENT_SETTINGS,&devmode);
        //return static_cast<float>(devmode.dmFields) / (info.rcMonitor.right - info.rcMonitor.left);
        return {devmode.dmPelsWidth,devmode.dmPelsHeight};
    }
};

static_assert(base::BitmapImplConcept<Bitmap>);
static_assert(base::ScreenRecorderImplConcept<ScreenRecorder,Bitmap>);

class ScreenBlocker : public base::ScreenBlocker<ScreenBlocker> {
public:
    ScreenBlocker() = delete;
    ScreenBlocker(ScreenBlocker&) = delete;
    ScreenBlocker(const base::Bitmap<win::Bitmap>& bitmap) {
        hbitmap = static_cast<HBITMAP>(CopyImage(static_cast<const win::Bitmap&>(bitmap).bitmap(),IMAGE_BITMAP,0,0,0));
        if(!hbitmap) {
            throw std::runtime_error("failed to copy bitmap");
        }
    };

    void show_impl() const {
        if(!msg_loop_thread.joinable())
            msg_loop_thread = std::thread(&ScreenBlocker::message_loop, this);
    }

    void hide_impl() const {
        if (hWndOverlay) {
            PostMessage(hWndOverlay,WM_CLOSE,0,0);
        }
        if(msg_loop_thread.joinable()) {
            msg_loop_thread.join();
        }
    }

private:
    std::thread* msg_thread;
    HBITMAP hbitmap;
    // mutable, 因为show_impl和hide_impl都是const
    mutable HWND hWndOverlay = nullptr;
    mutable std::thread msg_loop_thread;
    mutable std::atomic<bool> running{ false };

    void message_loop() const {
        HINSTANCE hInstance = GetModuleHandle(nullptr);
        static const wchar_t className[] = L"ScreenBlockerOverlayClass";

        // 注册窗口类
        WNDCLASSEX wc = {0};
        wc.cbSize = sizeof(WNDCLASSEX);
        wc.style = CS_HREDRAW | CS_VREDRAW;
        wc.lpfnWndProc = ScreenBlocker::WndProc;
        wc.hInstance = hInstance;
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.hbrBackground = reinterpret_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
        wc.lpszClassName = className;
        RegisterClassEx(&wc);

        // 创建全屏覆盖窗口（无边框，顶置，并且不在 Alt+Tab 中显示）
        hWndOverlay = CreateWindowEx(
            WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
            className,
            L"ScreenBlockerOverlay",
            WS_POPUP,
            0, 0,
            GetSystemMetrics(SM_CXSCREEN),
            GetSystemMetrics(SM_CYSCREEN),
            nullptr,
            nullptr,
            hInstance,
            (LPVOID)(this)  // 将 this 指针传给窗口过程
        );
        if (!hWndOverlay) {
            std::cerr << "Failed to create overlay window." << std::endl;
            return;
        }

        ShowWindow(hWndOverlay, SW_SHOW);
        UpdateWindow(hWndOverlay);

        running = true;
        MSG msg;
        while (GetMessage(&msg, nullptr, 0, 0) > 0) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        running = false;
        hWndOverlay = nullptr;
    }

    // 静态窗口过程：处理绘制、关闭等消息
    static LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
        ScreenBlocker* pThis = nullptr;
        if (uMsg == WM_NCCREATE) {
            CREATESTRUCT* cs = reinterpret_cast<CREATESTRUCT*>(lParam);
            pThis = reinterpret_cast<ScreenBlocker*>(cs->lpCreateParams);
            SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pThis));
        } else {
            pThis = reinterpret_cast<ScreenBlocker*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
        }

        switch (uMsg) {
        case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            
            // 设置高质量的拉伸模式
            SetStretchBltMode(hdc, HALFTONE);
            // 调整刷子原点（HALFTONE 模式下这是必要的）
            SetBrushOrgEx(hdc, 0, 0, nullptr);
        
            HDC hdcMem = CreateCompatibleDC(hdc);
            if (pThis) {
                HBITMAP hBmp = pThis->hbitmap;
                if (hBmp) {
                    HGDIOBJ oldBmp = SelectObject(hdcMem, hBmp);
                    BITMAP bmp;
                    GetObject(hBmp, sizeof(BITMAP), &bmp);
                    RECT rect;
                    GetClientRect(hwnd, &rect);
                    // 使用更高质量的模式拉伸位图填充整个窗口
                    StretchBlt(hdc, 0, 0, rect.right, rect.bottom,
                               hdcMem, 0, 0, bmp.bmWidth, bmp.bmHeight, SRCCOPY);
                    SelectObject(hdcMem, oldBmp);
                } else {
                    std::cerr << "Failed to get bitmap in WM_PAINT." << std::endl;
                }
            }
            DeleteDC(hdcMem);
            EndPaint(hwnd, &ps);
        }
        return 0;            
        case WM_DESTROY:
            // 收到WM_DESTROY时退出消息循环
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
        }
    }
};

static_assert(base::ScreenBlockerImpl<ScreenBlocker,Bitmap>);
}
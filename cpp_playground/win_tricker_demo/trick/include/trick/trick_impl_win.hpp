#include <trick/trick_base.hpp>
#define UNICODE
#include <windows.h>
#include <tchar.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <iostream>
#include <thread>
#include <random>

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

static_assert(base::BitmapImpl<Bitmap>);
static_assert(base::ScreenRecorderImpl<ScreenRecorder,Bitmap>);

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

    void show_impl() {
        if(!msg_loop_thread.joinable())
            msg_loop_thread = std::thread(&ScreenBlocker::message_loop, this);
    }

    void hide_impl() {
        if (hWndOverlay) {
            PostMessage(hWndOverlay,WM_CLOSE,0,0);
        }
        running = false;
        if(msg_loop_thread.joinable()) {
            msg_loop_thread.join();
        }
    }

private:
    std::thread* msg_thread;
    HBITMAP hbitmap;
    HWND hWndOverlay = nullptr;
    std::thread msg_loop_thread;
    std::atomic<bool> running{ false };

    void message_loop() {
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
        case WM_CLOSE:
            if(pThis->running)
                return 0;// block Alt + F4 and any other exit request from user input
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
        }
    }
};

static_assert(base::ScreenBlockerImpl<ScreenBlocker,Bitmap>);

class Beeper : public base::Beeper<Beeper> {
public:
    Beeper() = default;
    Beeper(Beeper&) = delete;
    Beeper(int freq) : freq(freq) {}

    ~Beeper() {
        random_beeping = false;
        if(beep_thread.joinable())
            beep_thread.join();
    }

    void beep_impl(int ms) const {
        Beep(freq,ms);
    }

    void start_random_beep_impl(int min_ms,int max_ms) {
        if(!beep_thread.joinable()) {
            random_beeping = true;
            beep_thread = std::thread(beep_task,this,min_ms,max_ms);
        }
    }

    void stop_random_beep_impl() {
        random_beeping = false;
    }

    void set_freq_impl(int freq) {
        if(freq < 0)
            throw std::invalid_argument("freq can't be negative");
        this->freq = freq;
    }

private:
    int freq{450};
    std::thread beep_thread;
    std::atomic<bool> random_beeping{false};

    void beep_task(int min_ms, int max_ms) {
        constexpr int MIN_BURST_COUNT = 1;              // 每次突发最少蜂鸣次数
        constexpr int MAX_BURST_COUNT = 8;              // 每次突发最多蜂鸣次数
        constexpr int INTRA_BURST_PAUSE_MIN_MS = 50;    // 蜂鸣之间最小暂停
        constexpr int INTRA_BURST_PAUSE_MAX_MS = 150;   // 蜂鸣之间最大暂停
        constexpr int BURST_PAUSE_MIN_MS = 500;         // 突发之间最小间隔
        constexpr int BURST_PAUSE_MAX_MS = 1000;        // 突发之间最大间隔
    
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> beep_dist(min_ms, max_ms);
        std::uniform_int_distribution<> burst_count_dist(MIN_BURST_COUNT, MAX_BURST_COUNT);
        std::uniform_int_distribution<> intra_burst_dist(INTRA_BURST_PAUSE_MIN_MS, INTRA_BURST_PAUSE_MAX_MS);
        std::uniform_int_distribution<> burst_pause_dist(BURST_PAUSE_MIN_MS, BURST_PAUSE_MAX_MS);
    
        int beep_duration, intra_pause, burst_pause, burst_count;
        while (random_beeping) {
            // 决定本次突发内的蜂鸣次数
            burst_count = burst_count_dist(gen);
            for (int i = 0; i < burst_count && random_beeping; ++i) {
                beep_duration = beep_dist(gen);
                // 调用蜂鸣函数
                beep_impl(beep_duration);
                // 如果不是最后一次蜂鸣，则短暂停顿
                if (i < burst_count - 1) {
                    intra_pause = intra_burst_dist(gen);
                    std::this_thread::sleep_for(std::chrono::milliseconds(intra_pause));
                }
            }
            // 突发后长时间等待，再开始下一组蜂鸣
            burst_pause = burst_pause_dist(gen);
            std::this_thread::sleep_for(std::chrono::milliseconds(burst_pause));
        }
    }    
};

static_assert(base::BeeperImpl<Beeper>);
}
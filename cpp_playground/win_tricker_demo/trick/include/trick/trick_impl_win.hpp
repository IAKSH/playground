#include <trick/trick_base.hpp>
#include <Windows.h>
#include <stdexcept>
#include <fstream>
#include <vector>

namespace trick::win {
class Bitmap : public trick::base::Bitmap<Bitmap> {
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

private:
    HBITMAP hbitmap;
};

class ScreenRecorder : public trick::base::ScreenRecorder<ScreenRecorder,Bitmap> {
public:
    ScreenRecorder() 
        : ScreenRecorder(GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN)) {}

    ScreenRecorder(int w,int h) : width(w), height(h) {
        screenDC = GetDC(nullptr);
        memoryDC = CreateCompatibleDC(screenDC);
        bitmap = CreateCompatibleBitmap(screenDC,width,height);
        SelectObject(memoryDC,bitmap);
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
};

static_assert(base::BitmapImplConcept<Bitmap>);
static_assert(base::ScreenRecorderImplConcept<ScreenRecorder,Bitmap>);
}
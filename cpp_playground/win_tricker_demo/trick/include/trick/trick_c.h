#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// 不透明类型
typedef struct trick_Bitmap trick_Bitmap;
typedef struct trick_ScreenRecorder trick_ScreenRecorder;

/* Bitmap 接口 */
int trick_bitmap_width(const trick_Bitmap* bitmap);
int trick_bitmap_height(const trick_Bitmap* bitmap);
void trick_bitmap_save_to_file(const trick_Bitmap* bitmap, const char* path);
void trick_bitmap_destroy(trick_Bitmap* bitmap);

/* ScreenRecorder 接口 */
trick_ScreenRecorder* trick_screen_recorder_create(int width, int height);
void trick_screen_recorder_destroy(trick_ScreenRecorder* recorder);
trick_Bitmap* trick_screen_recorder_capture(trick_ScreenRecorder* recorder);

#ifdef __cplusplus
}
#endif

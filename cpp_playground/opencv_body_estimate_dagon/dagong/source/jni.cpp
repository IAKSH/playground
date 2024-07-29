#include <jni.h>
#include <dagong/Demo.h>
#include <opencv2/opencv.hpp>

extern std::string body_estimate_demo(const std::string& base64_img);

std::string demo(const std::string& base64_img) {
    return body_estimate_demo(base64_img);
}

extern "C" JNIEXPORT jstring JNICALL
Java_Demo_demo(JNIEnv *env, jclass /* this */, jstring base64_img) {
    const char *nativeString = env->GetStringUTFChars(base64_img, 0);
    std::string result = demo(nativeString);
    env->ReleaseStringUTFChars(base64_img, nativeString);

    return env->NewStringUTF(result.c_str());
}
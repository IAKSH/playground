#include <stdio.h>
#include <string.h>

// strlen不统计'/0'

int main(void) {
    const char* str = "nihao";
    printf("strlen=%d\n",strlen(str));
    return 0;
}
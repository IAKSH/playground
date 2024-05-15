#include <stdio.h>

// 原理：通过按位与0x01将除了最低位外的所有位抹成0，只留下最后一位不断跳变，很明显偶0奇1
// 理论上应该比%2的方法更快

int main(void) {
    int n;
    while(1) {
        scanf("%d",&n);
        if(n <= 0)
            break;
        printf("%d & 1 = %d \t %s\n",n,(n & 1),(n & 1 ? "false" : "true"));   
    }
    return 0;
}
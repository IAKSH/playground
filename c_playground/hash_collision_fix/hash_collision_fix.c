#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 很烂，但是懒得改

int arr[9];
int input[] = {13,15,22,8,34,19,21,29};

int hash(int k) {
    int ret = (k % 7) + 1;
    printf("%d -> %d\n",k,ret);
    return ret;
}

void init() {
    memset(arr,9,0);
}

// di: 1 .. len
int fix_collision_liner(int hash_index,int di) {
    int ret = (hash_index + di) % 9;
    printf("\t %d -> %d (修正)\n",hash_index,ret);
    return ret;
} 

void insert_liner(int val) {
    int hash_index = hash(val);
    int* ori = arr + hash_index;
    if(!(*ori))
        *ori = val;
    else {
        int fixed_index;
        for(int i = 1;i < 8;++i) {
            fixed_index = fix_collision_liner(hash_index,i);
            if(!arr[fixed_index]) {
                arr[fixed_index] = val;
                return;
            }
        }
        puts("找不到空位\n");
    }
}

void test_liner() {
    for(int i = 0;i < sizeof(input) / sizeof(int);i++)
        insert_liner(input[i]);
}

// di: 1,-1,2,-2 .. len/2
int fix_collision_double(int hash_index, int di) {
    int ret;
    if(di > 0) 
        ret = (hash_index + di * di) % 9;
    else
        ret = (hash_index - di * di) % 9;
    printf("\t %d -> %d (修正)\n",hash_index,ret);
    if(ret < 0) {
        ret += 9;
        printf("\t\t -> %d (负值修正)\n",ret);
    }
    return ret;
}

void insert_double(int val) {
    int hash_index = hash(val);
    int* ori = arr + hash_index;
    if(!(*ori))
        *ori = val;
    else {
        int fixed_index;
        for(int i = 1;i < 8;++i) {
            fixed_index = fix_collision_double(hash_index,i);
            if(!arr[fixed_index]) {
                arr[fixed_index] = val;
                return;
            }
            fixed_index = fix_collision_double(hash_index,-i);
            if(!arr[fixed_index]) {
                arr[fixed_index] = val;
                return;
            }
        }
        puts("找不到空位\n");
    }
}

void test_double() {
    for(int i = 0;i < sizeof(input) / sizeof(int);i++)
        insert_double(input[i]);
}

int main(void) {
    //test_liner();
    test_double();
    return 0;
}
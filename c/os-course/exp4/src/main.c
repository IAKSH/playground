#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

#define FAT16_FREE_MARK 0x0000
#define FAT16_END_MARK 0xFFFF

enum FCB_TYPE {
    FCB_TYPE_FILE = 0x01,
    FCB_TYPE_DIR = 0x02
};

typedef struct {
    char name[16];
    uint32_t timestamp;
    uint32_t size;
    uint16_t first_block;
    uint8_t type;
} FCB;

typedef struct {
    char format[5];
    uint16_t data_block_num;
    uint16_t data_block_size;
} VDiskHeader;

typedef struct {
    // free_list_offset = 10
    // free_list_len (byte) = 4 * data_block_num
    // fat_offset = 10 + data_block_num * sizeof(int32_t) = 10 + data_block_num * 4
    // fat_len (byte) = 2 * data_block_num
    // data_offset = 10 + data_block_num * 4 + data_block_num * 2 = 10 + data_block_num * 6
    // data_len (byte) = data_block_num * data_block_size
    VDiskHeader header;
    int32_t* free_list;
    uint16_t* fat;
    //uint16_t* data_blocks;
    FILE* fp;
} VDisk;

VDisk init_vdisk(const char* path, uint16_t data_block_num, uint16_t data_block_size) {
    VDisk vdisk;
    if (access(path, F_OK) != -1) {
        vdisk.fp = fopen(path, "rb+");
        if (!vdisk.fp) {
            fprintf(stderr, "Can't open vdisk file for reading: %s\n", path);
            exit(1);
        }
        if (fread(&vdisk.header, sizeof(VDiskHeader), 1, vdisk.fp) != 1) {
            fprintf(stderr, "Error reading vdisk header\n");
            exit(1);
        }
        if (strcmp(vdisk.header.format, "VDISK") != 0) {
            fprintf(stderr, "Unknown file format: %s\n", vdisk.header.format);
            exit(1);
        }
        vdisk.free_list = (int32_t*)calloc(data_block_num, sizeof(int32_t));
        vdisk.fat = (uint16_t*)calloc(data_block_num, sizeof(uint16_t));
        if (!vdisk.free_list || !vdisk.fat) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        if (fread(vdisk.free_list, sizeof(int32_t), data_block_num, vdisk.fp) != data_block_num) {
            fprintf(stderr, "Error reading free_list\n");
            exit(1);
        }
        if (fread(vdisk.fat, sizeof(uint16_t), data_block_num, vdisk.fp) != data_block_num) {
            fprintf(stderr, "Error reading FAT\n");
            exit(1);
        }
    } else {
        strcpy(vdisk.header.format, "VDISK");
        vdisk.header.data_block_num = data_block_num;
        vdisk.header.data_block_size = data_block_size;

        vdisk.fp = fopen(path, "wb");
        if (!vdisk.fp) {
            fprintf(stderr, "Can't open vdisk file for writing: %s\n", path);
            exit(1);
        }
        if (fwrite(&vdisk.header, sizeof(VDiskHeader), 1, vdisk.fp) != 1) {
            fprintf(stderr, "Error writing vdisk header\n");
            exit(1);
        }

        vdisk.free_list = (int32_t*)calloc(data_block_num, sizeof(int32_t));
        vdisk.fat = (uint16_t*)calloc(data_block_num, sizeof(uint16_t));
        if (!vdisk.free_list || !vdisk.fat) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        // 初始化 free_list 和 fat
        for (int i = 0; i < data_block_num; i++) {
            vdisk.free_list[i] = data_block_num - i - 1;
            vdisk.fat[i] = FAT16_FREE_MARK;
        }
        vdisk.fat[0] = FAT16_END_MARK;
        vdisk.free_list[0] = -1;

        if (fwrite(vdisk.free_list, sizeof(int32_t), data_block_num, vdisk.fp) != data_block_num) {
            fprintf(stderr, "Error writing free_list\n");
            exit(1);
        }
        if (fwrite(vdisk.fat, sizeof(uint16_t), data_block_num, vdisk.fp) != data_block_num) {
            fprintf(stderr, "Error writing FAT\n");
            exit(1);
        }

        time_t t = time(NULL);
        FCB root_dir = { "/", t, 0, 0, FCB_TYPE_DIR };
        if (fwrite(&root_dir, sizeof(FCB), 1, vdisk.fp) != 1) {
            fprintf(stderr, "Error writing root directory FCB\n");
            exit(1);
        }
    }
    return vdisk;
}

void close_vdisk(VDisk* vdisk) {
    // 回写 free_list 和 fat
    fseek(vdisk->fp, 10, SEEK_SET);
    if (fwrite(vdisk->free_list, sizeof(int32_t), vdisk->header.data_block_num, vdisk->fp) != vdisk->header.data_block_num) {
        fprintf(stderr, "Error writing free_list to vdisk\n");
    }
    fseek(vdisk->fp, 10 + 4 * vdisk->header.data_block_num, SEEK_SET);
    if (fwrite(vdisk->fat, sizeof(uint16_t), vdisk->header.data_block_num, vdisk->fp) != vdisk->header.data_block_num) {
        fprintf(stderr, "Error writing fat to vdisk\n");
    }
    fclose(vdisk->fp);

    // 释放内存
    free(vdisk->free_list);
    free(vdisk->fat);
}

void create_empty_file(VDisk* vdisk,const char* name,uint32_t size) {
    int blockes_needed = ceil((float)size / vdisk->header.data_block_size) + 1;
    int i = 0;
    FCB fcb;
    for(;i < vdisk->header.data_block_num;i++) {
        if(vdisk->free_list[i] + 1 >= blockes_needed) {
            for(int j = 0;j < blockes_needed;j++)
                vdisk->free_list[i + j] = -1;
        
            int last_fat_i = 0;
            while(vdisk->fat[last_fat_i] != FAT16_END_MARK) {
                fseek(vdisk->fp,10 + 6 * vdisk->header.data_block_num + last_fat_i * vdisk->header.data_block_size,SEEK_SET);
                fread(&fcb,sizeof(FCB),1,vdisk->fp);
                if(strcmp(fcb.name,name) == 0) {
                    fputs("naming conflict\n",stderr);
                    exit(1);
                }
                last_fat_i = vdisk->fat[last_fat_i];
            }
            vdisk->fat[last_fat_i] = i;
            vdisk->fat[i] = FAT16_END_MARK;
 
            fcb.first_block = i;
            fcb.size = size;
            fcb.timestamp = time(NULL);
            fcb.type = FCB_TYPE_FILE;

            strcpy(fcb.name,name);
            fseek(vdisk->fp,10 + 6 * vdisk->header.data_block_num + fcb.first_block * vdisk->header.data_block_size,SEEK_SET);
            fwrite(&fcb,sizeof(FCB),1,vdisk->fp);
            return;
        }
    }
    fputs("no enough space\n",stderr);
    exit(1);
}

void create_subdir(VDisk* vdisk,const char* name) {
    int i = 0;
    FCB fcb;
    for(;i < vdisk->header.data_block_num;i++) {
        if(vdisk->free_list[i] >= 0) {
            vdisk->free_list[i] = -1;
        
            int last_fat_i = 0;
            while(vdisk->fat[last_fat_i] != FAT16_END_MARK) {
                fseek(vdisk->fp,10 + 6 * vdisk->header.data_block_num + last_fat_i * vdisk->header.data_block_size,SEEK_SET);
                fread(&fcb,sizeof(FCB),1,vdisk->fp);
                if(strcmp(fcb.name,name) == 0) {
                    fputs("naming conflict\n",stderr);
                    exit(1);
                }
                last_fat_i = vdisk->fat[last_fat_i];
            }
            vdisk->fat[last_fat_i] = i;
            vdisk->fat[i] = FAT16_END_MARK;
 
            fcb.first_block = i;
            fcb.size = 0;
            fcb.timestamp = time(NULL);
            fcb.type = FCB_TYPE_DIR;

            strcpy(fcb.name,name);
            fseek(vdisk->fp,10 + 6 * vdisk->header.data_block_num + fcb.first_block * vdisk->header.data_block_size,SEEK_SET);
            fwrite(&fcb,sizeof(FCB),1,vdisk->fp);
            return;
        }
    }
    fputs("no enough space\n",stderr);
    exit(1);
}

void delete_file(VDisk* vdisk,const char* name) {
    int last_fat_i,fat_i = 0;
    FCB fcb;
    while(vdisk->fat[fat_i] != FAT16_END_MARK) {
        fseek(vdisk->fp,10 + 6 * vdisk->header.data_block_num + fat_i * vdisk->header.data_block_size,SEEK_SET);
        fread(&fcb,sizeof(FCB),1,vdisk->fp);
        if(fcb.type == FCB_TYPE_FILE && strcmp(fcb.name,name) == 0) {
            int blocks = ceil((float)fcb.size / vdisk->header.data_block_size) + 1;
            for(int j = 0;j < blocks;j++) {
                if(fcb.first_block + blocks - j < vdisk->header.data_block_num)
                    vdisk->free_list[fcb.first_block + blocks - 1 - j] = vdisk->free_list[fcb.first_block + blocks - j] + 1;
                else
                    vdisk->free_list[fcb.first_block + blocks - 1 - j] = 0;
            }
            vdisk->fat[last_fat_i] = vdisk->fat[fat_i];
            vdisk->fat[fat_i] = FAT16_FREE_MARK;
            return;
        }
        last_fat_i = fat_i;
        fat_i = vdisk->fat[fat_i];
    }
    fputs("file not found\n",stderr);
    exit(1);
}

void delete_subdir(VDisk* vdisk,const char* name) {
    int last_fat_i,fat_i = 0;
    FCB fcb;
    while(vdisk->fat[fat_i] != FAT16_END_MARK) {
        fseek(vdisk->fp,10 + 6 * vdisk->header.data_block_num + fat_i * vdisk->header.data_block_size,SEEK_SET);
        fread(&fcb,sizeof(FCB),1,vdisk->fp);
        if(fcb.type == FCB_TYPE_DIR && strcmp(fcb.name,name) == 0) {
            if(fcb.first_block + 1 < vdisk->header.data_block_num)
                vdisk->free_list[fcb.first_block] = vdisk->free_list[fcb.first_block + 1] + 1;
            else
                vdisk->free_list[fcb.first_block] = 0;
            vdisk->fat[last_fat_i] = vdisk->fat[fat_i];
            vdisk->fat[fat_i] = FAT16_FREE_MARK;
            return;
        }
        last_fat_i = fat_i;
        fat_i = vdisk->fat[fat_i];
    }
    fputs("file not found\n",stderr);
    exit(1);
}

#include <stdio.h>
#include <time.h>

void list_dir(VDisk* vdisk) {
    int fat_i = 0;
    FCB fcb;
    while(true) {
        fseek(vdisk->fp, 10 + 6 * vdisk->header.data_block_num + fat_i * vdisk->header.data_block_size, SEEK_SET);
        fread(&fcb, sizeof(FCB), 1, vdisk->fp);

        // 转换 timestamp
        time_t raw_time = fcb.timestamp;
        struct tm *time_info;
        char time_str[20];

        time_info = localtime(&raw_time);
        strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", time_info);

        fprintf(stdout, "%s\t%s\t%s\t%dByte\n",
            fcb.type == FCB_TYPE_FILE ? "FILE" : "DIR",
            fcb.name,
            time_str,
            fcb.size
        );

        if (vdisk->fat[fat_i] != FAT16_END_MARK)
            fat_i = vdisk->fat[fat_i];
        else
            break;
    }
}

void process_command(VDisk* vdisk, char* command) {
    char cmd[256];
    char arg1[256];
    char arg2[256];

    int count = sscanf(command, "%s %s %[^\n]", cmd, arg1, arg2);

    if (count == 1 && strcmp(cmd, "SAV") == 0) {
        // 保存当前的vdisk
        close_vdisk(vdisk);
        *vdisk = init_vdisk("vdisk.img",32,1024);
        printf("vdisk saved.\n");
    } else if (count == 1 && strcmp(cmd, "EXT") == 0) {
        // 退出程序
        close_vdisk(vdisk);
        printf("Exiting...\n");
        exit(0);
    } else if (count == 2 && strcmp(cmd, "MD") == 0) {
        create_subdir(vdisk, arg1);
    } else if (count == 2 && strcmp(cmd, "RD") == 0) {
        delete_subdir(vdisk, arg1);
    } else if (count == 3 && strcmp(cmd, "MK") == 0) {
        create_empty_file(vdisk, arg1, atoi(arg2));
    } else if (count == 2 && strcmp(cmd, "DEL") == 0) {
        delete_file(vdisk, arg1);
    } else if (count == 1 && strcmp(cmd, "LS") == 0) {
        list_dir(vdisk);
    } else {
        printf("Unknown command: %s\n", command);
    }
}

#define FAT16_MAX_LEN UINT16_MAX
#define FAT16_BLOCK_SIZE 1024

int main() {
    VDisk vdisk = init_vdisk("vdisk.img",32,32);

    char command[512];    

    while (1) {
        printf("$ ");
        if (fgets(command, sizeof(command), stdin) != NULL) {
            command[strcspn(command, "\n")] = '\0';  // 去除换行符
            process_command(&vdisk, command);
        }
    }

    close_vdisk(&vdisk);
    return 0;
}
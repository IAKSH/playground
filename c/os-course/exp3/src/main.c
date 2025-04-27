#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    STATE_RUNNING,
    STATE_READY,
    STATE_BLOCKED
} State;

typedef struct {
    char id[16];
    State state;
} PCB;

typedef struct __ProcessNode {
    PCB* pcb;
    struct __ProcessNode* next;
    struct __ProcessNode* pre;
} ProccessNode;

typedef struct __DeviceNode {
    char id[16];
    State state;
    PCB* occupying_process;
    ProccessNode* waiting_head;
    ProccessNode* waiting_tail;
    struct __DeviceNode* upper;
    struct __DeviceNode* next;
} DeviceNode;

typedef DeviceNode DCTable;
typedef DeviceNode COCTable;
typedef DeviceNode CHCTable;

typedef enum {
    DEVICE_TYPE_MOUSE,
    DEVICE_TYPE_KEYBOARD,
    DEVICE_TYPE_PRINTER,
    DEVICE_TYPE_DISPLAY,
    DEVICE_TYPE_OTHERS
} DeviceType;

typedef struct __SDTNode {
    DeviceType device_type;
    DeviceNode* node;
    struct __SDTNode* next;
} SDTNode;

typedef SDTNode SDTable;

SDTable sdt_head;
DCTable dct_head;
COCTable coct_head;
CHCTable chct_head;

void add_device_node(DeviceNode* head, DeviceNode* new_node) {
    DeviceNode* node = head;
    while (node->next)
        node = node->next;
    node->next = new_node;
}

void add_sdt_ndoe(DCTable* new_dct, DeviceType type) {
    SDTNode* new_sdt_node = (SDTNode*)malloc(sizeof(SDTNode));
    new_sdt_node->device_type = type;
    new_sdt_node->node = new_dct;
    new_sdt_node->next = NULL;

    SDTNode* node = &sdt_head;
    while (node->next)
        node = node->next;
    node->next = new_sdt_node;
}

DeviceNode* create_device_node(const char* name) {
    DeviceNode* node = (DeviceNode*)malloc(sizeof(DeviceNode));
    strcpy(node->id, name);
    node->state = STATE_READY;
    node->occupying_process = NULL;
    node->waiting_head = NULL;
    node->waiting_tail = NULL;
    node->upper = NULL;
    node->next = NULL;
    return node;
}

CHCTable* create_channel(const char* name) {
    DeviceNode* node = create_device_node(name);
    add_device_node(&chct_head, node);
    return node;
}

COCTable* create_controller(const char* name, CHCTable* channel) {
    DeviceNode* node = create_device_node(name);
    node->upper = channel;
    add_device_node(&coct_head, node);
    return node;
}

DCTable* create_device(const char* name, COCTable* controller, DeviceType type) {
    DeviceNode* node = create_device_node(name);
    node->upper = controller;
    add_device_node(&dct_head, node);
    add_sdt_ndoe(node, type);
    return node;
}

void show_all_sdt() {
    SDTNode* node = sdt_head.next;
    while (node) {
        const char* type_str;
        switch (node->device_type) {
        case DEVICE_TYPE_PRINTER:
            type_str = "printer";
            break;
        case DEVICE_TYPE_MOUSE:
            type_str = "mouse";
            break;
        case DEVICE_TYPE_KEYBOARD:
            type_str = "keyboard";
            break;
        case DEVICE_TYPE_DISPLAY:
            type_str = "display";
            break;
        default:
            type_str = "others";
        }

        const char* state_str;
        switch (node->node->state) {
        case STATE_RUNNING:
            state_str = "running";
            break;
        case STATE_READY:
            state_str = "ready";
            break;
        case STATE_BLOCKED:
            state_str = "blocked";
            break;
        default:
            state_str = "unknown";
        }

        printf("/dev/%s\ttype: %s\tstate: %s\n", node->node->id, type_str, state_str);
        node = node->next;
    }
}

void show_all_device_node(DeviceNode* head, const char* prefix) {
    DeviceNode* node = head->next;
    while (node) {
        const char* state_str;
        switch (node->state) {
        case STATE_RUNNING:
            state_str = "running";
            break;
        case STATE_READY:
            state_str = "ready";
            break;
        case STATE_BLOCKED:
            state_str = "blocked";
            break;
        default:
            state_str = "unknown";
        }

        printf("/%s/%s\tstate: %s\n", prefix, node->id, state_str);
        node = node->next;
    }
}

void delete_device_node(DeviceNode* head, const char* id) {
    DeviceNode* node = head;
    DeviceNode* prev = NULL;
    while (node && strcmp(node->id, id) != 0) {
        prev = node;
        node = node->next;
    }
    if (node) {
        if (prev) {
            prev->next = node->next;
        } else {
            head->next = node->next;
        }
        free(node);
    }
}

void delete_channel(const char* id) {
    delete_device_node(&chct_head, id);
}

void delete_controller(const char* id) {
    delete_device_node(&coct_head, id);
}

void delete_device(const char* id) {
    delete_device_node(&dct_head, id);
}

void allocate_device(PCB* process, const char* device_id) {
    DeviceNode* device = dct_head.next;
    while (device && strcmp(device->id, device_id) != 0) {
        device = device->next;
    }
    if (device) {
        if (device->occupying_process == NULL) {
            device->occupying_process = process;
            process->state = STATE_BLOCKED;
            printf("Device %s allocated to process %s\n", device_id, process->id);
        } else {
            process->state = STATE_BLOCKED;
            ProccessNode* new_waiting_node = (ProccessNode*)malloc(sizeof(ProccessNode));
            new_waiting_node->pcb = process;
            new_waiting_node->next = NULL;
            new_waiting_node->pre = NULL;
            if (device->waiting_tail) {
                device->waiting_tail->next = new_waiting_node;
                new_waiting_node->pre = device->waiting_tail;
                device->waiting_tail = new_waiting_node;
            } else {
                device->waiting_head = new_waiting_node;
                device->waiting_tail = new_waiting_node;
            }
            printf("Device %s is occupied, process %s added to waiting queue\n", device_id, process->id);
        }
    }
}

void release_device(PCB* process, const char* device_id) {
    DeviceNode* device = dct_head.next;
    while (device && strcmp(device->id, device_id) != 0) {
        device = device->next;
    }
    if (device && device->occupying_process == process) {
        device->occupying_process = NULL;
        process->state = STATE_READY;
        printf("Device %s released by process %s\n", device_id, process->id);
        if (device->waiting_head) {
            ProccessNode* waiting_node = device->waiting_head;
            device->occupying_process = waiting_node->pcb;
            waiting_node->pcb->state = STATE_BLOCKED;
            device->waiting_head = waiting_node->next;
            if (device->waiting_head == NULL) {
                device->waiting_tail = NULL;
            } else {
                device->waiting_head->pre = NULL;
            }
            free(waiting_node);
            printf("Device %s allocated to waiting process %s\n", device_id, device->occupying_process->id);
        }
    }
}

void process_command(char* command, PCB* process1, PCB* process2, PCB* process3) {
    char cmd[16];
    char proc_id[16];
    sscanf(command, "%s %s", cmd, proc_id);
    PCB* process = NULL;
    if (strcmp(proc_id, "p1") == 0) {
        process = process1;
    } else if (strcmp(proc_id, "p2") == 0) {
        process = process2;
    } else if (strcmp(proc_id, "p3") == 0) {
        process = process3;
    }

    if (process == NULL) {
        printf("Invalid process ID\n");
        return;
    }

    if (strcmp(cmd, "alloc") == 0) {
        char device_id[16];
        sscanf(command + strlen(cmd) + strlen(proc_id) + 2, "%s", device_id);
        allocate_device(process, device_id);
    } else if (strcmp(cmd, "release") == 0) {
        char device_id[16];
        sscanf(command + strlen(cmd) + strlen(proc_id) + 2, "%s", device_id);
        release_device(process, device_id);
    } else {
        printf("Unknown command\n");
    }
}

int main() {
    sdt_head.next = NULL;
    dct_head.next = NULL;
    coct_head.next = NULL;
    chct_head.next = NULL;

    CHCTable* channel_1 = create_channel("channel_1");
    CHCTable* channel_2 = create_channel("channel_2");
    COCTable* controller_1 = create_controller("controller_1", channel_1);
    COCTable* controller_2 = create_controller("controller_2", channel_1);
    COCTable* controller_3 = create_controller("controller_3", channel_2);
    DCTable* keyboard_1 = create_device("keyboard_1", controller_1, DEVICE_TYPE_KEYBOARD);
    DCTable* mouse_1 = create_device("mouse_1", controller_1, DEVICE_TYPE_MOUSE);
    DCTable* printer_1 = create_device("printer_1", controller_1, DEVICE_TYPE_PRINTER);
    DCTable* display_1 = create_device("display_1", controller_2, DEVICE_TYPE_DISPLAY);

    show_all_sdt();
    printf("--------------\n");
    show_all_device_node(&chct_head, "cha");
    printf("--------------\n");
    show_all_device_node(&coct_head, "con");
    printf("--------------\n");
    show_all_device_node(&dct_head, "dev");

    PCB pcb_1 = {"p1", STATE_READY};
    PCB pcb_2 = {"p2", STATE_READY};
    PCB pcb_3 = {"p3", STATE_READY};

    char command[64];
    while (1) {
        printf("Enter command (alloc/release <process_id> <device_id>): ");
        if (fgets(command, sizeof(command), stdin) == NULL) {
            break;
        }
        if (strncmp(command, "exit", 4) == 0) {
            break;
        }
        process_command(command, &pcb_1, &pcb_2, &pcb_3);
    }

    return 0;
}

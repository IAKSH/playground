#pragma once

// 读取和分析姿态
void task_posture(void);
// 维持姿态
void task_stabilizer(void);
// 摄像头编解码
void task_camera(void);
// 监测电池电压
void task_power(void);
// 状态灯
void task_led(void);
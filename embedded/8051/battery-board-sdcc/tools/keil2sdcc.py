#!/usr/bin/env python3
import re
import sys

def process_line(line, sfr_dict):
    """
    对一行进行判断与转换：
      - 如果是 sfr 定义，转换为 __sfr __at(addr) Name;
      - 如果是 sbit 定义，则查找对应的 sfr 地址并计算地址，转换为 __sbit __at(addr) Name;
      - 如果是 #define 定义的 xdata 寄存器，则调整类型顺序和添加 __xdata；
      - 其它行保持不变。
    """
    # 1. 处理 sfr 定义：
    # 匹配例子：   sfr P0 = 0x80;
    m = re.match(r'^\s*sfr\s+(\w+)\s*=\s*(0x[0-9A-Fa-f]+)\s*;', line)
    if m:
        name, addr = m.groups()
        sfr_dict[name] = addr  # 存储该 SFR 的地址，供 sbit 使用
        return f"__sfr __at({addr}) {name};\n"

    # 2. 处理 sbit 定义：
    # 匹配例子：   sbit P00 = P0^0;
    m = re.match(r'^\s*sbit\s+(\w+)\s*=\s*(\w+)\s*\^\s*(\d+)\s*;', line)
    if m:
        name, base, bit = m.groups()
        try:
            base_addr = int(sfr_dict.get(base, "0"), 16)
            bit_index = int(bit)
            sbit_addr = hex(base_addr + bit_index)
        except Exception as e:
            sbit_addr = f"{base}^{bit}"  # 若发生异常则原样输出
        return f"__sbit __at({sbit_addr}) {name};\n"

    # 3. 处理 #define 定义的 XDATA 类型寄存器：
    # 匹配例子：   #define PWM0C (*(unsigned int volatile xdata *)0xff00)
    # 可能有多重空格或数据类型顺序不同，我们使用非贪婪匹配。
    m = re.match(r'^\s*#define\s+(\w+)\s+\(\*\(\s*(.*?)\s+xdata\s*\*\)\s*(0x[0-9A-Fa-f]+)\s*\)', line)
    if m:
        name = m.group(1)
        type_text = m.group(2).strip()
        addr = m.group(3)
        # 如果 type_text 中已经有 volatile，则将其去掉再统一放到前面
        if "volatile" in type_text:
            type_text = re.sub(r'\bvolatile\b', '', type_text).strip()
            type_str = f"volatile {type_text}"
        else:
            type_str = type_text
        # 输出时使用 SDCC 常用的写法：添加 __xdata 关键字
        return f"#define {name} (*(volatile {type_text} __xdata *){addr})\n"

    # 其它情况返回原始行
    return line

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_c51_to_sdcc.py input_file.h > output_file.h", file=sys.stderr)
        sys.exit(1)
    input_filename = sys.argv[1]
    sfr_dict = {}
    with open(input_filename, 'r') as infile:
        for line in infile:
            sys.stdout.write(process_line(line, sfr_dict))

if __name__ == '__main__':
    main()

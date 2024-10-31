        AREA EXAMPLE,CODE,READONLY	;定义本段名称和属性
        ENTRY						;程序入口
        MOV R1,#1
LOOP	ADD R0,R0,R1                ;R0 = R0 + R1
		CMP R0,#3
		BLS LOOP                    ;如果R0 < 3则跳转到LOOP
		END
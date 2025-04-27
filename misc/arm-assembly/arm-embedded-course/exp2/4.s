N EQU 5                             ;定义常量N = 5
        AREA EXAMPLE,CODE,READONLY	
        ENTRY
     	CODE32
START
		LDR R0,=N                   ;加载N到R0
		MOV R2,R0                   ;R2 = R0
		MOV R0,#0                   ;R0 = 0
		MOV R1,#0                   ;R1 = 0
LOOP
		CMP R1,R2
		BHI END_LOOP                ;R1 > R2则跳转到END_LOOP
		ADD R0,R0,R1                ;R0 = R0 + R1
		ADD R1,R1,#1                ;R1 = R1 + 1
		B LOOP				        ;跳转到LOOP
END_LOOP
        END